import pytest

from app.agents.synthesizer import _build_synthesis_request, synthesize
from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from app.core.errors import NoAvailableModelError
from tests.helpers import build_test_engine


def _node(id_: str, status: str, content: str | None = None, error: str | None = None) -> DagNodeResult:
    response = None
    if content is not None:
        response = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": content})])
    return DagNodeResult(id=id_, status=status, response=response, error=error)


def _dag(nodes: list[DagNodeResult], status: str = "success") -> DagRunResponse:
    return DagRunResponse(id="dag_test", status=status, nodes=nodes, latency_ms=0.0)


# --- pure request-building ---------------------------------------------------

def test_build_synthesis_request_includes_task_and_step_results():
    dag = _dag([_node("a", "success", content="Paris")])
    req = _build_synthesis_request("what's the capital of France?", None, dag, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "what's the capital of France?" in joined
    assert "Paris" in joined


def test_build_synthesis_request_includes_context_when_given():
    dag = _dag([_node("a", "success", content="Paris")])
    req = _build_synthesis_request("task", "the user is a beginner", dag, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the user is a beginner" in joined


# --- end-to-end via a real ChatEngine (FakeProvider-backed) ------------------

@pytest.mark.asyncio
async def test_synthesize_composes_one_answer_from_multiple_nodes(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "Paris, translated as Madrid's counterpart."}})
    dag = _dag([_node("capital", "success", content="Paris"), _node("translate", "success", content="Madrid")])

    response = await synthesize(engine, "tell me France's capital, translated", None, dag)

    from app.contracts.response import extract_message_text
    assert extract_message_text(response).strip() == "Paris, translated as Madrid's counterpart."


@pytest.mark.asyncio
async def test_synthesize_propagates_no_available_model_error(tmp_path):
    engine = await build_test_engine(tmp_path, {})  # no providers registered at all
    dag = _dag([_node("a", "success", content="x")])

    with pytest.raises(NoAvailableModelError):
        await synthesize(engine, "task", None, dag)
