from collections.abc import Iterator
import contextlib
import pathlib

import pytest

from squill.repo import Repository
from squill.repo import CycleError, HeadError, ReadError, SequenceError


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> Iterator[Repository]:
    with contextlib.chdir(tmp_path):
        yield Repository(pathlib.Path('repo'))


def assert_persisted(expected: Repository) -> None:
    """Assert that the expected repository is persisted and that reading the
    repository again yields the same revision graph.
    """
    repo = Repository(expected.path)

    assert repo.heads == expected.heads

    for head in repo.heads:
        assert repo.sequence(target=head) == expected.sequence(target=head)


@contextlib.contextmanager
def assert_unmodified(repo: Repository) -> Iterator[None]:
    """Context manager for asserting that a repository is not modified in the
    block.
    """
    revs_before = {h: repo.sequence(target=h) for h in repo.heads}

    yield

    revs_after = {h: repo.sequence(target=h) for h in repo.heads}

    assert revs_after == revs_before
    assert_persisted(repo)


def test_empty(repo: Repository) -> None:
    assert repo.heads == set()
    assert repo.head is None
    assert repo.sequence() == []


def test_add_revisions(repo: Repository) -> None:
    r0 = repo.add()

    assert r0
    assert repo.heads == {r0}
    assert repo.head == r0
    assert repo.sequence() == [r0]

    assert (repo.path / r0 / 'deploy.sql').read_text() == ""
    assert (repo.path / r0 / 'revert.sql').read_text() == ""

    r1 = repo.add(parent=r0)

    assert r1
    assert repo.heads == {r1}
    assert repo.head == r1
    assert repo.sequence() == [r0, r1]

    r2 = repo.add(parent=r1)

    assert r2
    assert repo.heads == {r2}
    assert repo.head == r2
    assert repo.sequence() == [r0, r1, r2]

    assert_persisted(repo)


def test_add_duplicate_revision_key(repo: Repository) -> None:
    r0 = repo.add()

    with pytest.raises(ValueError) as excinfo:
        repo.add(key=r0)

    assert str(excinfo.value) == f"duplicate revision {r0!r}"


def test_add_with_unknown_parent(repo: Repository) -> None:
    with pytest.raises(ValueError) as excinfo:
        repo.add(parent='foo')

    assert str(excinfo.value) == "unknown parent 'foo'"


def test_hydra(repo: Repository) -> None:
    head0 = repo.add()
    head1 = repo.add()

    assert repo.heads == {head0, head1}

    with pytest.raises(HeadError) as excinfo:
        repo.head

    assert excinfo.value.heads == {head0, head1}

    with pytest.raises(HeadError) as excinfo:
        repo.sequence()

    assert excinfo.value.heads == {head0, head1}

    assert_persisted(repo)


def test_revisions_with_explicit_base(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    assert repo.sequence(base=r1) == [r1, r2]


def test_revisions_unreachable_base(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    # Revision key that is guaranteed to not exist.
    norev = r0 + r1 + r2

    with pytest.raises(SequenceError) as excinfo:
        repo.sequence(base=norev)

    assert excinfo.value.target == r2
    assert excinfo.value.base == norev


def test_revisions_unreachable_target(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    with pytest.raises(SequenceError) as excinfo:
        repo.sequence(base=r2, target=r1)

    assert excinfo.value.target == r1
    assert excinfo.value.base == r2


def test_rebase_on_head(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)
    r3 = repo.add(parent=r0)
    r4 = repo.add(parent=r3)

    assert repo.heads == {r2, r4}

    repo.rebase(key=r3, parent=r2)

    assert repo.heads == {r4}
    assert repo.head == r4
    assert repo.sequence() == [r0, r1, r2, r3, r4]

    assert_persisted(repo)


def test_rebase_not_on_head(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    with (
        assert_unmodified(repo),
        pytest.raises(ValueError) as excinfo,
    ):
        repo.rebase(key=r2, parent=r0)

    assert str(excinfo.value) == f"new parent {r0!r} must be a current head"


def test_rebase_cycle(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    with (
        assert_unmodified(repo),
        pytest.raises(CycleError) as excinfo,
    ):
        repo.rebase(key=r1, parent=r2)

    assert excinfo.value.revisions == [r1, r2]


def test_rebase_onto_self(repo: Repository) -> None:
    r0 = repo.add()

    with (
        assert_unmodified(repo),
        pytest.raises(CycleError) as excinfo,
    ):
        repo.rebase(key=r0, parent=r0)

    assert excinfo.value.revisions == [r0]


def test_read_malformed_line(repo: Repository) -> None:
    r0 = repo.add()
    path = repo.path / r0 / 'revision'

    with path.open('a') as fp:
        fp.write("xxx\n")

    with pytest.raises(ReadError) as excinfo:
        Repository(repo.path)

    assert str(excinfo.value) == rf"malformed line: 'xxx\n' ({path}:1)"


def test_read_duplicate_property(repo: Repository) -> None:
    r0 = repo.add()
    path = repo.path / r0 / 'revision'

    with path.open('a') as fp:
        fp.write("foo: bar\nfoo: bar\n")

    with pytest.raises(ReadError) as excinfo:
        Repository(repo.path)

    assert str(excinfo.value) == f"duplicate property: 'foo' ({path}:2)"


def test_read_no_head(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)

    # create cycle r0->r1->r0->...
    (repo.path / r0 / 'revision').write_text(f"Parent: {r1}\n")

    # reload repository
    repo = Repository(repo.path)

    assert repo.heads == set()
    assert repo.sequence() == []

    with pytest.raises(CycleError) as excinfo:
        repo.sequence(target=r1)

    assert excinfo.value.revisions == [r0, r1]


def test_read_cycle(repo: Repository) -> None:
    r0 = repo.add()
    r1 = repo.add(parent=r0)
    r2 = repo.add(parent=r1)

    # create cycle r0->r1->r0->...
    (repo.path / r0 / 'revision').write_text(f"Parent: {r1}\n")

    # reload repository
    repo = Repository(repo.path)

    assert repo.heads == {r2}

    with pytest.raises(CycleError) as excinfo:
        repo.sequence()

    assert excinfo.value.revisions == [r0, r1]
