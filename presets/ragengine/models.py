# Copyright (c) KAITO authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any

from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    doc_id: str = Field(default="")
    text: str
    metadata: dict | None = Field(default_factory=dict)
    hash_value: str | None = None
    is_truncated: bool = False


class ListDocumentsResponse(BaseModel):
    documents: list[Document]  # List of DocumentResponses
    count: int  # Number of documents in the current response


class IndexRequest(BaseModel):
    index_name: str
    documents: list[Document]


class UpdateDocumentRequest(BaseModel):
    documents: list[Document]


class UpdateDocumentResponse(BaseModel):
    updated_documents: list[Document]
    unchanged_documents: list[Document]
    not_found_documents: list[Document]


class DeleteDocumentRequest(BaseModel):
    doc_ids: list[str]


class DeleteDocumentResponse(BaseModel):
    deleted_doc_ids: list[str]
    not_found_doc_ids: list[str]


class QueryRequest(BaseModel):
    index_name: str
    query: str
    top_k: int = 5
    # Accept a dictionary for our LLM parameters
    llm_params: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional parameters for the language model, e.g., temperature, top_p",
    )
    # Accept a dictionary for rerank parameters
    rerank_params: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Experimental: Optional parameters for reranking. Only 'top_n' and 'choice_batch_size' are supported.",
    )

    @model_validator(mode="after")
    def validate_params(cls, values: "QueryRequest") -> "QueryRequest":
        # Access fields as attributes instead of treating as a dictionary
        rerank_params = values.rerank_params
        top_k = values.top_k

        # Validate rerank parameters
        if "top_n" in rerank_params:
            if not isinstance(rerank_params["top_n"], int):
                raise ValueError("Invalid type: 'top_n' must be an integer.")
            if rerank_params["top_n"] > top_k:
                raise ValueError(
                    "Invalid configuration: 'top_n' for reranking cannot exceed 'top_k' from the RAG query."
                )

        return values


# Define models for NodeWithScore, and QueryResponse
class NodeWithScore(BaseModel):
    doc_id: str
    node_id: str
    text: str
    score: float
    metadata: dict | None = None


class QueryResponse(BaseModel):
    response: str
    source_nodes: list[NodeWithScore]
    metadata: dict | None = None


class HealthStatus(BaseModel):
    status: str
    detail: str | None = None
