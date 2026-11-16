# Written for the CausalDiagnosis fork of AssetOpsBench, not part of IBM/ReActXen.
"""Stand-ins for colorama's Fore and Style, whose escape codes the vendored agents print around every line: every
attribute is the empty string, so the printout of an episode is plain text and the fork needs no colorama."""


class _Blank:
    def __getattr__(self, name):
        return ""


Fore = _Blank()
Style = _Blank()
