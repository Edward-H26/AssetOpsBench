"""Every place in the files of the package that names a section of one of its Markdown pages, as "PAGE.md section N",
"PAGE.md, section NAME" or "section N of `PAGE.md`", names a heading of that page, so that a renamed heading or a moved section
fails here. A numbered section matches the number before the first dot of a heading, and a named section the start of a heading,
without regard to case."""
import re
from pathlib import Path

from CausalDiagnosis.experiments.run_verification.upload_manifest import shipped
from CausalDiagnosis.utils.paths import ROOT

FORMS = (re.compile(r"(CausalDiagnosis/[\w./-]+\.md),?\s+section\s+([0-9]+|[A-Z][a-z]+(?:\s+[a-z]+)*)"),
         re.compile(r"section\s+([0-9]+)\s+of\s+`(CausalDiagnosis/[\w./-]+\.md)`"))
SUFFIXES = (".py", ".md", ".sh", ".sbatch")  # the files whose prose names a section: the code, the pages and the job scripts


def section_references(root):
    """[(file, page, section)] of every place in the files of the package below root, as an upload ships them, that names a
    section of a page. The text of a file is read with its line breaks folded, so that a name split over two lines reads whole."""
    places = []
    for name in shipped(root):
        if not name.startswith("CausalDiagnosis/") or not name.endswith(SUFFIXES):
            continue
        text = " ".join((Path(root) / name).read_text(encoding="utf-8", errors="replace").split())
        for number, form in enumerate(FORMS):
            for match in form.finditer(text):
                page, section = (match.group(1), match.group(2)) if number == 0 else (match.group(2), match.group(1))
                places.append((name, page, section))
    return places


def names_a_heading(page, section):
    """Whether the page has a heading of that number or that name."""
    lines = page.read_text(encoding="utf-8").splitlines() if page.is_file() else []
    headings = [line.lstrip("#").strip() for line in lines if line.startswith("#")]
    if section.isdigit():
        return any(heading.split(".")[0] == section for heading in headings)
    return any(heading.lower().startswith(section.lower()) for heading in headings)


def missing(root):
    """The places below root whose section is no heading of its page, one line each."""
    return [f"{name}: {page} section {section}" for name, page, section in section_references(root)
            if not names_a_heading(Path(root) / page, section)]


def test_every_section_that_a_file_of_the_package_names_is_a_heading_of_its_page():
    assert section_references(ROOT)  # the forms match places of the tree, so the check below reads something
    assert missing(ROOT) == []


def test_a_section_without_its_heading_is_named_and_the_kept_ones_pass(tmp_path):
    page = "CausalDiagnosis/docs/page.md"  # named through a variable, so that this module holds no place of its own
    (tmp_path / page).parent.mkdir(parents=True)
    (tmp_path / page).write_text("# Page\n\n## Kept heading\n\nText.\n\n## 2. Second part\n", encoding="utf-8")
    (tmp_path / "CausalDiagnosis" / "module.py").write_text(
        f'"""See {page}, section Kept heading, and section 2 of `{page}`. The rule is in\n'
        f'{page} section Gone part. Its count is in {page} section 3."""\n', encoding="utf-8")
    assert len(section_references(tmp_path)) == 4
    assert missing(tmp_path) == [f"CausalDiagnosis/module.py: {page} section Gone part", f"CausalDiagnosis/module.py: {page} section 3"]
