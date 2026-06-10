"""[DEPRECATED v6] Education parser — superseded by image_content_parser.

Keep this module for backward compatibility only. New code should use:
  from src.milvus.image_content_parser import ImageContentParser

EducationParser is now an alias for ImageContentParser.
"""

from src.milvus.image_content_parser import ImageContentParser as EducationParser  # noqa: F401

# Also re-export build_semantic_text for backward compat
build_semantic_text = EducationParser.build_semantic_text

__all__ = ["EducationParser", "build_semantic_text"]
