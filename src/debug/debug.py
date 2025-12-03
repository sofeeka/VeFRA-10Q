class DebugData_Document:
    def __init__(self, quarter: str, year: str):
        self.quarter = quarter
        self.year = year


class DebugData_Chunk:
    def __init__(self, text: str, document: DebugData_Document):
        self.text = text
        self.document = document


class DebugData_ExpandedQuery:
    def __init__(self, query: str, chunks: list[DebugData_Chunk]):
        self.query = query
        self.chunks = chunks


class DebugData:
    def __init__(self):
        self.user: str = ""
        self.question: str = ""
        self.reworded_question: str = ""
        self.answer: str = ""
        self.documents: list[DebugData_Document] = []
        self.unique_retrieved_chunks: list[DebugData_Chunk] = []
        self.expanded_queries: list[DebugData_ExpandedQuery] = []
        self.retrieved_chunks_per_query: dict[str, list[DebugData_Chunk]] = {}
        self.final_retrieved_chunks: list[str] = []
        self.system_prompt: str = ""
        self.use_prompt: str = ""
        self.duration: float = 0.0
        self.input_price: float = 0.0
        self.output_price: float = 0.0

    @property
    def price_per_1000_requests(self) -> float:
        return round((self.input_price + self.output_price) * 1000, 2)

    def add_document(self, quarter: str, year: str):
        document = DebugData_Document(
            quarter=quarter,
            year=year,
        )
        self.documents.append(document)

    def add_unique_retrieved_chunk(self, text: str, quarter: str, year: str):
        chunk = DebugData_Chunk(
            text=text,
            document=DebugData_Document(
                quarter=quarter,
                year=year,
            ),
        )
        self.unique_retrieved_chunks.append(chunk)

    def add_expanded_query(self, query: str, chunks: list[DebugData_Chunk]):
        self.expanded_queries.append(
            DebugData_ExpandedQuery(
                query=query,
                chunks=chunks,
            )
        )


class DebugManager:
    def __init__(self):
        self._debug_mode = False
        self._debug_data = DebugData()

    def enable(self):
        self._debug_mode = True
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
