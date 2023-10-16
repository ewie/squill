"""Module for handling repositories of database revisions.

The repository organizes database revisions in a tree to encode dependencies
between individual revisions.  Those dependencies emerge naturally when using
revisions to transform a database schema and/or its data from one state to
another.  The revisions must be applied in the right order to establish the
desired state in a database.  The repository provides a sequence of revisions
that transform a pristine database to any state.

When multiple developers work on one database schema it is inevitable that
two or more developers will base their work on the same revision.  The result
is multiple branches (hence the repository's tree structure) with revisions
that are not necessarily conflict-free.  The branches must first be integrated
to establish a unique sequence of revisions that can eventually be applied.
This is done by rebasing the revisions of one branch on another branch.
Conflicting revisions must be fixed in this process.

Every revision consist of a deploy and a revert script.  The deploy script
transforms a database to the desired state.  The revert script restores the
original state.  Schema changes can always be reverted but not necessarily
without data loss! Both scripts are written in the database systems's native
script language to be executed by that database system's standard client.

The revision tree is stored in the file system as plain text files to be
tracked in version control.
"""

from collections.abc import Iterator, Sequence
import pathlib
import re
import secrets
import typing as t


class RepositoryError(Exception):
    """Base class for repository errors.
    """


class CycleError(RepositoryError):
    """Raised when revisions form a cycle.
    """

    def __init__(self, revisions: Sequence[str]) -> None:
        self.revisions = revisions


class HeadError(RepositoryError):
    """Raised when the existence of multiple head revisions prevent
    a repository operation.
    """

    def __init__(self, heads: set[str]) -> None:
        self.heads = heads


class ReadError(RepositoryError):
    """Raised when reading a repository state fails.
    """

    def __init__(self, msg: str, path: pathlib.Path, lineno: int):
        self.msg = msg
        self.path = path
        self.lineno = lineno

    def __str__(self) -> str:
        return f"{self.msg} ({self.path}:{self.lineno})"


class SequenceError(RepositoryError):
    """Raised when there is no sequence of revisions from base to target.
    """

    def __init__(self, base: str, target: str) -> None:
        self.base = base
        self.target = target


class _Revision(t.NamedTuple):
    """Revision metadata.
    """

    key: str
    parent: str | None


_REVISION_FILENAME = 'revision'
"""Filename of revision metadata.
"""


class Repository:
    """Repository of database revisions.
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Open a repository at the given path and read all revisions.

        :param path: repository path
        :raise ReadError: if reading the repository fails
        """
        self._path = path
        self._revs = {r.key: r for r in self._read_all()}

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def heads(self) -> set[str]:
        """Return all head revisions.

        Head revisions are all revisions that are not also parent revisions.
        """
        parents = {r.parent for r in self._revs.values() if r.parent}
        return self._revs.keys() - parents

    @property
    def head(self) -> str | None:
        """Return the only head revision.

        :raise HeadError: if there are multiple heads
        """
        heads = self.heads

        if len(heads) > 1:
            raise HeadError(heads)

        for head in heads:
            return head

        return None

    def sequence(
        self, *,
        base: str | None = None,
        target: str | None = None,
    ) -> Sequence[str]:
        """Return the sequence of revisions between base and target (both
        inclusive).

        Both base and target revision are optional.  The root revision is used
        as default base.  The current head is used as the default target.

        :param base: optional base revision
        :param target: optional target revision
        :return: sequence of revisions from base to target (both inclusive)
        :raise CycleError: if revisions form a cycle
        ;raise HeadError: if no target is specified and there are multiple
            heads
        :raise SequenceError: if target cannot be reached through base
        """
        # Use head as default target.
        target = target or self.head

        # Collect revisions in reverse, between target and base.
        seq: list[str] = []

        # The current revision.
        key = target

        # Follow parents until reaching the base or detecting a cycle.
        while key:
            if key in seq:
                # Omit revisions that we collected but which are not part of
                # the detected cycle.
                seq = seq[seq.index(key):]
                raise CycleError(list(reversed(seq)))

            seq.append(key)

            if base and key == base:
                break  # reached the base

            key = self._revs[key].parent

        # Check if the reached base is the specified one.
        if seq and base and seq[-1] != base:
            assert target
            raise SequenceError(base, target)

        # Return the revisions between base and target.
        return list(reversed(seq))

    def add(
        self,
        key: str | None = None,
        parent: str | None = None,
    ) -> str:
        """Add a new revision with optional parent.

        A random revision key is generated if none is provided.

        :param key: optional key of new revision
        :param parent: optional parent of new revision
        :return: key of new revision
        :raise ValueError: on duplicate revision key or if parent revision
            does not exist
        """
        key = key or self._random_key()
        rev = _Revision(key=key, parent=parent)

        if rev.key in self._revs:
            raise ValueError(f"duplicate revision {rev.key!r}")

        if rev.parent and rev.parent not in self._revs:
            raise ValueError(f"unknown parent {rev.parent!r}")

        # Create the revision scripts.
        (self._path / rev.key).mkdir(parents=True)
        (self._path / rev.key / 'deploy.sql').touch()
        (self._path / rev.key / 'revert.sql').touch()

        self._write(rev)

        return rev.key

    def rebase(self, key: str, parent: str) -> None:
        """Rebase a revision on a head revision.

        Rebasing is used to remove branches and establish a revision sequence
        that can be applied to a database.  Rebasing cannot introduce any
        cycles.

        :param key: key of revision to rebase
        :param parent: new parent revision to rebase onto
        :raise CycleError: if rebasing would result in a revision cycle
        :raise ValueError: if `parent` is not a current head revision
        """
        if parent not in self.heads:
            raise ValueError(f"new parent {parent!r} must be a current head")

        # Prevent creation of cycle, i.e. rebasing on itself or any ancestor.
        # The new parent must already be reachable from the revision before
        # rebasing if the dependency on the new parent would cause a cycle.
        try:
            if cycle := self.sequence(base=key, target=parent):
                raise CycleError(cycle)
        except SequenceError:
            pass

        rev = self._revs[key]._replace(parent=parent)

        self._write(rev)

    def _random_key(self) -> str:
        return secrets.token_hex(6)

    def _read_all(self) -> Iterator[_Revision]:
        return map(self._read, self._path.glob(f'*/{_REVISION_FILENAME}'))

    def _read(self, path: pathlib.Path) -> _Revision:
        props = {}

        with path.open() as fp:
            for lineno, line in enumerate(fp, start=1):
                match = re.match(r'^([^:]+):\s+(\S+)$', line)

                if not match:
                    raise ReadError(
                        f"malformed line: {line!r}", path, lineno,
                    )

                key, val = match.groups()

                # Enforce unambiguous properties.
                if key in props:
                    raise ReadError(
                        f"duplicate property: {key!r}", path, lineno,
                    )

                props[key] = val

        return _Revision(
            key=path.parent.name,
            parent=props.get('Parent'),
        )

    def _write(self, rev: _Revision) -> None:
        with (self._path / rev.key / _REVISION_FILENAME).open('w') as fp:
            if rev.parent:
                fp.write(f"Parent: {rev.parent}\n")

        self._revs[rev.key] = rev
