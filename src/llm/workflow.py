from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.llm.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT
from src.models import TicketCategory


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None

    retriever: object
    session_store: object
    ticket_repo: object


class IntentClassification(BaseModel):
    intent: Literal["answer", "ticket"] = Field(
        description="Must be 'answer' for general questions, factual queries, trivia, or if they just want information. Must be 'ticket' ONLY if they report an unresolved personal issue or ask to file a ticket."
    )
    customer_name: str | None = Field(default=None, description="Customer name if provided")
    customer_email: str | None = Field(default=None, description="Customer email if provided")
    issue_description: str | None = Field(default=None, description="Issue description if provided")
    category: str | None = Field(default=None, description="Ticket category if identifiable: order, payment, account, technical, or other")


def build_support_workflow(model: BaseChatModel):
    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        retriever = state.get("retriever")
        query = state.get("customer_message", "")
        chunks: list[dict[str, str]] = []
        sources: list[str] = []

        if retriever and query:
            raw_results = await retriever.search(query)
            seen = set()
            for chunk in raw_results:
                key = (chunk["source"], chunk["content"])
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
                    if chunk["source"] not in sources:
                        sources.append(chunk["source"])

        return {"retrieved_chunks": chunks, "sources": sources}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        session_store = state.get("session_store")
        session_id = state.get("session_id", "")
        conversation_state = None
        if session_store and session_id:
            conversation_state = session_store.get_or_create(session_id)

        ticket_in_progress = False
        if conversation_state:
            if conversation_state.ticket_id:
                ticket_in_progress = True
            elif any(v is not None for v in [
                conversation_state.customer_name,
                conversation_state.customer_email,
                conversation_state.issue_description,
                conversation_state.category,
            ]):
                ticket_in_progress = True

        chunks = state.get("retrieved_chunks", [])
        context = "\n\n".join(c["content"] for c in chunks) if chunks else "No relevant context found."
        message = state.get("customer_message", "")

        history_text = ""
        if conversation_state and conversation_state.history:
            recent = conversation_state.history[-10:]
            history_text = "\n".join(f"{h['role']}: {h['content']}" for h in recent)

        session_info = ""
        if conversation_state:
            collected = []
            if conversation_state.customer_name:
                collected.append(f"name={conversation_state.customer_name}")
            if conversation_state.customer_email:
                collected.append(f"email={conversation_state.customer_email}")
            if conversation_state.issue_description:
                collected.append(f"issue={conversation_state.issue_description}")
            if conversation_state.category:
                collected.append(f"category={conversation_state.category}")
            if conversation_state.ticket_id:
                collected.append(f"ticket_id={conversation_state.ticket_id}")
            if collected:
                session_info = "Already collected: " + ", ".join(collected)

        classify_prompt = f"""You must classify the customer's intent into exactly one of these two categories: "answer" or "ticket".

"answer": The customer is asking ANY question, including policy questions, general information, or trivia (e.g., "What is the return policy?", "Who is the CEO?", "What is the CEO's phone number?", "How much is shipping?").
"ticket": The customer is reporting a specific problem, issue, or complaint (e.g., "My screen is broken", "I was charged twice", "I want to file a complaint").

Knowledge context:
{context}

Conversation history:
{history_text}

Session state:
{session_info}
Ticket collection in progress: {ticket_in_progress}

Customer message:
{message}

Rules:
1. If ticket collection is in progress (ticket_id exists), always classify as "ticket".
2. If the message is a factual question or trivia (like asking for a phone number or name), classify as "answer".
3. Extract customer_name, customer_email, issue_description, or category ONLY if explicitly provided by the user. Do not invent values."""

        try:
            structured_model = model.with_structured_output(IntentClassification)
            result = await structured_model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=classify_prompt),
            ])
        except Exception:
            if ticket_in_progress:
                return {"route": "ticket", "extracted_fields": {}}
            return {"route": "answer", "extracted_fields": {}}

        extracted: dict[str, str] = {}
        if result.customer_name:
            extracted["customer_name"] = result.customer_name
        if result.customer_email:
            extracted["customer_email"] = result.customer_email
        if result.issue_description:
            extracted["issue_description"] = result.issue_description
        if result.category and result.category in ("order", "payment", "account", "technical", "other"):
            extracted["category"] = result.category

        route = result.intent
        if ticket_in_progress and route == "answer":
            route = "ticket"

        return {"route": route, "extracted_fields": extracted}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        print("DEBUG: Executing ANSWER node")
        chunks = state.get("retrieved_chunks", [])
        message = state.get("customer_message", "")

        session_store = state.get("session_store")
        session_id = state.get("session_id", "")
        session_text = ""
        if session_store and session_id:
            conv = session_store.get_or_create(session_id)
            if conv.history:
                recent = conv.history[-6:]
                session_text = "\n".join(f"{h['role']}: {h['content']}" for h in recent)

        if chunks:
            context = "\n\n".join(c["content"] for c in chunks)
            sources = list(dict.fromkeys(c["source"] for c in chunks))
        else:
            context = "No relevant information found in the knowledge base."
            sources = []

        user_content = ANSWER_TEMPLATE.format(
            context=context,
            session=session_text or "New conversation",
            message=message,
        )

        if not chunks:
            user_content += "\n\nIMPORTANT: The knowledge base does not contain relevant information for this question. Clearly tell the customer that you don't have information about this topic and offer to open a support ticket if they need further help. Do NOT make up an answer."

        response = await model.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        return {
            "response_text": response.content,
            "sources": sources,
            "ticket_id": state.get("ticket_id"),
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        session_store = state.get("session_store")
        session_id = state.get("session_id", "")
        ticket_repo = state.get("ticket_repo")

        if not session_store or not session_id or not ticket_repo:
            return {
                "response_text": "I'm sorry, there was a system error. Please try again.",
                "sources": [],
                "ticket_id": None,
            }

        conv = session_store.get_or_create(session_id)

        if conv.ticket_id:
            return {
                "response_text": f"Your support ticket has already been created with ID {conv.ticket_id}. Our team will follow up with you soon. Is there anything else I can help you with?",
                "sources": [],
                "ticket_id": conv.ticket_id,
            }

        extracted = state.get("extracted_fields", {})

        if "customer_name" in extracted and extracted["customer_name"]:
            conv.customer_name = extracted["customer_name"]
        if "customer_email" in extracted and extracted["customer_email"]:
            conv.customer_email = extracted["customer_email"]
        if "issue_description" in extracted and extracted["issue_description"]:
            conv.issue_description = extracted["issue_description"]
        if "category" in extracted and extracted["category"]:
            if extracted["category"] in ("order", "payment", "account", "technical", "other"):
                conv.category = extracted["category"]

        missing = conv.missing_ticket_fields()

        if missing:
            field_prompts = {
                "customer_name": "Could you please provide your full name?",
                "customer_email": "Could you please provide your email address so we can follow up with you?",
                "issue_description": "Could you please describe the issue you're experiencing in detail?",
                "category": "What category best describes your issue? Options are: order, payment, account, technical, or other.",
            }
            next_field = missing[0]
            prompt_text = field_prompts.get(next_field, f"Could you please provide your {next_field.replace('_', ' ')}?")

            already_collected = []
            if conv.customer_name:
                already_collected.append(f"Name: {conv.customer_name}")
            if conv.customer_email:
                already_collected.append(f"Email: {conv.customer_email}")
            if conv.issue_description:
                already_collected.append(f"Issue: {conv.issue_description}")
            if conv.category:
                already_collected.append(f"Category: {conv.category}")

            response = f"I'd be happy to help you create a support ticket. {prompt_text}"
            if already_collected:
                response = f"Thank you! I have the following information so far:\n" + "\n".join(f"- {item}" for item in already_collected) + f"\n\n{prompt_text}"

            return {
                "response_text": response,
                "sources": [],
                "ticket_id": None,
            }

        from src.tools.ticket_tool import create_ticket_tool
        tool = create_ticket_tool(ticket_repo, session_id)

        summary = conv.issue_description[:155] + "..." if len(conv.issue_description) > 155 else conv.issue_description
        if len(summary) < 5:
            summary = f"Support request: {summary}"

        ticket_id = tool.invoke({
            "customer_name": conv.customer_name,
            "customer_email": conv.customer_email,
            "issue_description": conv.issue_description,
            "category": conv.category,
            "summary": summary,
        })

        conv.ticket_id = ticket_id

        return {
            "response_text": f"Your support ticket has been created successfully! Your ticket ID is {ticket_id}. Our team will review your case and follow up at {conv.customer_email}. Is there anything else I can help you with?",
            "sources": [],
            "ticket_id": ticket_id,
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route", "")
        if route not in ("answer", "ticket"):
            return "answer"
        return route

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()
