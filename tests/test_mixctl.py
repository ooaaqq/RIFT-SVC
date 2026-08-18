from __future__ import annotations

from pathlib import Path

import pytest

from rift_svc.mixctl import (
    PROJECT_DIRS,
    SHELF_DIRS,
    copy_project_file,
    create_audition,
    create_project,
    create_version,
    create_workspace,
)


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    workspace = create_workspace(tmp_path / "露早")
    project = create_project(workspace, 13, "很抱歉我这么可爱", "小雾uya")
    return workspace, project


def test_workspace_is_a_sorted_human_shelf(tmp_path: Path) -> None:
    workspace = create_workspace(tmp_path / "露早")

    assert [path.name for path in sorted(workspace.iterdir())] == list(SHELF_DIRS)
    assert not (workspace / "library").exists()
    assert not (workspace / "workspace.toml").exists()


def test_project_uses_official_title_cover_credit_and_readable_sections(
    tmp_path: Path,
) -> None:
    _, project = make_project(tmp_path)

    assert project.name == "013. 很抱歉我这么可爱-小雾uya"
    assert all((project / name).is_dir() for name in PROJECT_DIRS)
    assert not (project / "NOTES.md").exists()
    assert not (project / "assets/manifest.json").exists()


def test_copy_creates_an_ordinary_self_contained_file(tmp_path: Path) -> None:
    _, project = make_project(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF-audio-placeholder")

    copied = copy_project_file(
        project,
        source,
        Path("10. Sources/Demo Song-Demo Singer.wav"),
    )

    assert copied.read_bytes() == source.read_bytes()
    assert copied.is_file()
    assert not copied.is_symlink()


def test_copy_refuses_overwrite_and_parent_escape(tmp_path: Path) -> None:
    _, project = make_project(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    destination = Path("10. Sources/source.wav")
    copy_project_file(project, source, destination)

    with pytest.raises(FileExistsError):
        copy_project_file(project, source, destination)
    with pytest.raises(ValueError):
        copy_project_file(project, source, Path("../outside.wav"))


def test_listen_copies_readable_regular_files(tmp_path: Path) -> None:
    _, project = make_project(tmp_path)
    source = tmp_path / "candidate.wav"
    source.write_bytes(b"candidate")

    created = create_audition(project, [("12. 小雾uya - RIFT Direct", source)])

    assert created == [project / "00. Listen/12. 小雾uya - RIFT Direct.wav"]
    assert created[0].read_bytes() == b"candidate"
    assert not created[0].is_symlink()


def test_version_name_explains_the_change_and_can_snapshot_files(tmp_path: Path) -> None:
    _, project = make_project(tmp_path)
    mix = tmp_path / "mix.wav"
    script = tmp_path / "render.sh"
    mix.write_bytes(b"mix")
    script.write_text("#!/bin/sh\n")

    version = create_version(
        project,
        5,
        "Wider and Louder Harmony",
        [("Mix.wav", mix), ("Scripts/render.sh", script)],
    )

    assert version.name == "05. Wider and Louder Harmony"
    assert (version / "Mix.wav").read_bytes() == b"mix"
    assert (version / "Scripts/render.sh").is_file()


def test_failed_version_is_removed_atomically(tmp_path: Path) -> None:
    _, project = make_project(tmp_path)
    source = tmp_path / "mix.wav"
    source.write_bytes(b"mix")

    with pytest.raises(FileNotFoundError):
        create_version(
            project,
            6,
            "Incomplete Render",
            [("Mix.wav", source), ("Missing.wav", tmp_path / "missing.wav")],
        )

    versions = project / "40. Versions"
    assert not (versions / "06. Incomplete Render").exists()
    assert list(versions.iterdir()) == []


def test_project_and_version_numbers_are_bounded(tmp_path: Path) -> None:
    workspace = create_workspace(tmp_path / "露早")

    with pytest.raises(ValueError):
        create_project(workspace, 1000, "Song", "Singer")
    project = create_project(workspace, 1, "Song", "Singer")
    with pytest.raises(ValueError):
        create_version(project, 100, "Too High", [])
