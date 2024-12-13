import markdown_to_json as mdj
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
)

from llama_index.core import (
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage,
)

from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from typing import Union, List
from llama_index.core.node_parser import SentenceSplitter


CITATION_TEMPLATE = PromptTemplate("""You are a thesis supervisor at a university. You have access to a collection of academic papers and books covering the topics of cultural diplomacy in the Gulf, Middle Eastern politics.
You are tasked with helping a student prepare his thesis defence to the best of your ability. To this end, you will need to answer his questions with the utmost accuracy, using primarily the documents available to you.
If the question contains multiple parts, you must make sure to address every single one expletively and accurately using information from the knowledgebase.
If the documents retrieved contain quantitative metrics (e.g: statistics, tabular information) relevant to the question, you must include them.
You are encouraged to combine and compare information from all of your knowledgebases to compose your answer, as long as you stay accurate.

Once you have scoured your personal databases:
- You MUST include as much ACCURATE context as you can about history, geography, cultural evolution, and politics as is helpful.
- You MUST synthesize the information in a coherent way
- You MUST cite your sources in line, and provide links when available.
- You MUST provide an analysis of any numerical data you are using.
- You MUST provide a clear answer, and make suggestions for further research when helpful or necessary.
- After using the provided documents, you can then use your own knowledge as a thesis defender, and the vastness of the internet to contextualize your response.

You MUST cite your sources. Your answers must be long and detailed, and you must provide as much context as possible. You must answere in a legible, structured format.

If none of your source documents hold the answer, you should make that clear. Finally, you must look at the users original question and the final answer you generated. Very precisely identify what information you think is missing, and suggest the user further exploratory questions \n
The user can ask follow up questions to you and you want to guide the user to information you feel was missed and should be added from the perspective of a thesis supervisor. You must include a citation for every point provided from the source documents.

For example:
Query: {query_string}?
Answer:

{context_str}""")

CITATION_TEMPLATE_REFINE = PromptTemplate("""You are a thesis supervisor at a university. You have access to a collection of academic papers and books covering the topics of cultural diplomacy in the Gulf, Middle Eastern politics.
You are tasked with helping a student prepare his thesis defence to the best of your ability. To this end, you will need to answer his questions with the utmost accuracy, using primarily verified information from reputable sources with a prioritization of your personal database.
If the question contains multiple parts, you must make sure to address every single one expletively and accurately using information from the knowledgebase.
If the

Once you have scoured your personal databases:
- You MUST include as much ACCURATE context as you can about history, geography, cultural evolution, and politics as is helpful.
- You MUST synthesize the information in a coherent way
- You MUST cite your sources in line, and provide links when available.
- You MUST provide an analysis of any numerical data you are using.
- You MUST provide a clear answer, and make suggestions for further research when helpful or necessary.
- You can then use your own knowledge and experience as a thesis defender to contextualize your response.

If none of your source documents hold the answer, you should make that clear. Finally, you must look at the users original question and the final answer you generated. Very precisely identify what information you think is missing, and suggest the user further exploratory questions \n
The user can ask follow up questions to you and you want to guide the user to information you feel was missed and should be added from the perspective of a thesis supervisor. 
You must include inline citations for information provided from the source documents in the form (Document Name). You must give your response in a structured and readable format, and you must use markdown headings (#) to separate different sections of your response. You MUST NOT use bold or italics in your response.

You have already been given an answer. Now you must refine it!
For example:
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Query: When is water wet?
Answer: Water will be wet when the sky is red [Author],
which occurs in the evening [Author].
Now it's your turn.
We have provided an existing answer: {existing_answer}
Below are several numbered sources of information.
Use them to refine the existing answer.
If the provided sources are not helpful, you will repeat the existing answer.
Begin refining!
{context_msg} \n
Query: {query_str}
Answer: 

{source_titles}
""")

app = FastAPI()
storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./VectorStore"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./VectorStore"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./VectorStore"),
    )
data_index = load_index_from_storage(storage_context)

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class CreateCitationsEvent(Event):
    """Add citations to the nodes."""
    nodes: list[NodeWithScore]

class CitationQueryEngineWorkflow(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> Union[RetrieverEvent, None]:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if ev.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=ev.get("n_sources", 5))
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def create_citation_nodes(
        self, ev: RetrieverEvent
    ) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        nodes = ev.nodes
        new_nodes: List[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=750,
            chunk_overlap=100,
        )

        for node in nodes:
            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )
            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes)+1}: {node.dict()['node']['relationships']['1']['metadata']['title']}\n{text_chunk}\n"
                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )
                new_node.node.text = text
                new_nodes.append(new_node)
        print(f"Created citations.")
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a streaming response using the retrieved nodes."""
        llm = OpenAI(model="gpt-4o", timeout=60)
        query = await ctx.get("query", default=None)
        print(f"Synthesizing response for query: {query}")
        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=CITATION_TEMPLATE,
            refine_template=CITATION_TEMPLATE_REFINE,
            response_mode=ResponseMode.REFINE,
            use_async=True,
        )
        print("Awaiting response")
        result = await synthesizer.asynthesize(query, nodes=ev.nodes)
        print(f"Done {result}!")
        return StopEvent(result=result)

# Initialize your index here (e.g., loading documents and creating an index)
# This is an example setup; customize as needed.


async def run_workflow(query, number_of_sources):
        w = CitationQueryEngineWorkflow(timeout=600)
        response = await w.run(query=query, index=data_index, n_sources=number_of_sources)
        return response

@app.post('/query')
async def handle_query(
    query: str = Query(..., description="The question or topic you want to query."),
    number_of_sources: int = Query(5, description="The number of sources desired, defaults to 5. The more sources, the more detailed the response, but the longer it takes. For reference, 5 sources takes about 30 seconds.")
):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is missing")
    
    # Run the async workflow within an event loop
    result_event = await run_workflow(query, number_of_sources)
    if result_event is None:
        raise HTTPException(status_code=404, detail="No results found")
    print(result_event)
    print("")
    print(mdj.jsonify(result_event.response))
    print(mdj.dictify(result_event.response))
    print(result_event.get_formatted_sources())

    return JSONResponse(content=mdj.dictify(result_event.response))

@app.get('/')
async def index():
    return RedirectResponse(url="/docs")