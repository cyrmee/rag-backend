class UnsupportedFileType(Exception):
    pass


class DuplicateDocument(Exception):
    """The file's bytes are identical to a document already ingested under
    another filename."""

    def __init__(self, filename: str, existing_filename: str):
        super().__init__(f"'{filename}' is identical to the already uploaded '{existing_filename}'")
        self.filename = filename
        self.existing_filename = existing_filename
