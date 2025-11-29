class DebugData_Document:
    quarter: str
    year: str

    def __init__(self):
        self.quarter = ""
        self.year = ""


class DebugData_Chunk:
    text: str
    document: DebugData_Document

    def __init__(self):
        self.text = ""
        self.document = DebugData_Document()


class DebugData_ExpandedQuery:
    def __init__(self, query: str):
        self.query = query
        # self.chunks: list[str] = []


class DebugData:
    def __init__(self):
        self.user: str = ""
        self.question: str = ""
        self.answer: str = ""
        self.documents: list[DebugData_Document] = []
        self.chunks: list[DebugData_Chunk] = []
        self.expanded_queries: list[DebugData_ExpandedQuery] = []

    def add_document(self, quarter, year):
        document = DebugData_Document()
        document.quarter = quarter
        document.year = year

        self.documents.append(document)

    def add_chunk(self, text, quarter, year):
        chunk = DebugData_Chunk()
        chunk.text = text
        chunk.document.quarter = quarter
        chunk.document.year = year

        self.chunks.append(chunk)

    def add_expanded_query(self, query):
        self.expanded_queries.append(DebugData_ExpandedQuery(query=query))


class DebugManager:
    _debug_mode: bool = False
    _debug_data: DebugData = None

    def __init__(self):
        self._debug_mode = False
        self._debug_data = DebugData()

    def enable(self):
        self._debug_mode = True
        # Reset data when enabling? Or keep it?
        # The original code created new data on get_debug_data if None.
        # Let's ensure we have a fresh data object if we are enabling.
        if self._debug_data is None:
            self._debug_data = DebugData()

    def disable(self):
        self._debug_mode = False

    def is_enabled(self) -> bool:
        return self._debug_mode

    def get_data(self) -> DebugData:
        return self._debug_data

    def clear_data(self):
        self._debug_data = DebugData()


debug_manager = DebugManager()
