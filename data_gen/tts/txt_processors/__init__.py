"""Text processor package.

Avoid importing optional language-specific dependencies at package import time.
Processors register lazily via get_txt_processor_cls(...).
"""
