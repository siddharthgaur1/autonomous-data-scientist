"""Streamlit dashboard. Talks to the API over HTTP — no shared imports, no shared DB.

Keeping it to the public API is what makes it a real second service: if the
dashboard needs something the API doesn't expose, that's a missing endpoint, not
a reason to reach into SQLite behind the API's back.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 3

STAGES = [
    ("supervisor", "Understanding the goal"),
    ("cleaning", "Cleaning the data"),
    ("eda", "Exploring"),
    ("features", "Engineering features"),
    ("model_selection", "Comparing models"),
    ("tuning", "Tuning"),
    ("evaluation", "Evaluating"),
    ("narrative", "Writing it up"),
    ("report", "Building the deck"),
    ("reviewer", "Reviewing"),
]

st.set_page_config(page_title="Autonomous Data Scientist", page_icon="🧪", layout="wide")


def api_get(path: str, **kwargs):
    try:
        response = requests.get(f"{API}{path}", timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def artifact_url(run_id: str, path: str) -> str:
    return f"{API}/runs/{run_id}/artifacts/{path}"


def download_artifact(run_id: str, path: str) -> bytes | None:
    try:
        response = requests.get(artifact_url(run_id, path), timeout=60)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def render_progress(run: dict) -> None:
    """Show which agent is working right now."""
    current = run.get("current_stage")
    status = run.get("status")

    done_stages = []
    for name, _ in STAGES:
        done_stages.append(name)
        if name == current:
            break

    completed = status == "completed"
    fraction = 1.0 if completed else len(done_stages) / (len(STAGES) + 1)
    st.progress(fraction)

    columns = st.columns(len(STAGES))
    for column, (name, label) in zip(columns, STAGES):
        if completed or name in done_stages[:-1]:
            column.markdown(f"✅ **{label}**")
        elif name == current:
            column.markdown(f"⏳ **{label}**")
        else:
            column.markdown(f"<span style='color:#898781'>{label}</span>",
                            unsafe_allow_html=True)


def render_escalation(run: dict) -> None:
    """The agent stopped and wants a human. Make that unmissable."""
    st.warning("This run paused and needs your input rather than guessing.", icon="🙋")
    st.markdown(f"**The agent asks:**\n\n{run.get('human_question')}")
    with st.form("escalation"):
        answer = st.text_area("Your answer", height=100)
        if st.form_submit_button("Send and resume") and answer.strip():
            try:
                requests.post(
                    f"{API}/runs/{run['run_id']}/answer",
                    json={"answer": answer},
                    timeout=30,
                ).raise_for_status()
                st.success("Sent. The run is resuming.")
                time.sleep(1)
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Could not resume: {exc}")


def render_results(run: dict) -> None:
    """Findings, charts, metrics, downloads."""
    evaluation = run.get("eval_metrics") or {}
    metrics = evaluation.get("metrics") or {}

    if metrics:
        st.subheader("How well it works")
        columns = st.columns(min(len(metrics), 5))
        for column, (name, value) in zip(columns, metrics.items()):
            column.metric(name, f"{value:.4f}")
        if evaluation.get("split"):
            st.caption(evaluation["split"])
        if evaluation.get("interpretation"):
            st.info(evaluation["interpretation"])

    if run.get("narrative"):
        st.subheader("What we found")
        st.write(run["narrative"])

    plots = [a for a in run.get("artifacts", []) if a["kind"] == "plot"]
    if plots:
        st.subheader("Charts")
        for left, right in zip(plots[::2], plots[1::2] + [None]):
            columns = st.columns(2)
            for column, plot in zip(columns, [left, right]):
                if not plot:
                    continue
                data = download_artifact(run["run_id"], plot["path"])
                if not data:
                    continue
                with column:
                    if plot["path"].endswith(".png"):
                        st.image(data, caption=plot["label"], use_container_width=True)
                    else:
                        st.components.v1.html(data.decode("utf-8"), height=420)

    candidates = run.get("candidate_models") or []
    if candidates:
        st.subheader("Models compared")
        st.dataframe(
            [
                {
                    "Model": c["name"],
                    c["metric"]: c["baseline_score"],
                    "Chosen": "✓" if c["name"] == run.get("chosen_model") else "",
                    "Notes": c["notes"],
                }
                for c in candidates
            ],
            use_container_width=True,
            hide_index=True,
        )

    tuning = run.get("tuning_results")
    if tuning and tuning.get("n_trials"):
        st.caption(
            f"Optuna: {tuning['n_trials']} trials · "
            f"{tuning['baseline_score']} → {tuning['best_score']} · "
            f"best params: {tuning['best_params']}"
        )

    transformations = run.get("transformations") or []
    features = run.get("engineered_features") or []
    if transformations or features:
        with st.expander("What the agent did to the data"):
            if transformations:
                st.markdown("**Cleaning**")
                st.dataframe(transformations, use_container_width=True, hide_index=True)
            if features:
                st.markdown("**Features**")
                st.dataframe(features, use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    columns = st.columns(3)
    for column, kind, label, filename in [
        (columns[0], "report", "📊 Download report (.pptx)", "report.pptx"),
        (columns[1], "model", "🧠 Download model (.pkl)", "model.pkl"),
        (columns[2], "model_card", "📇 Download model card (.json)", "model_card.json"),
    ]:
        artifact = next((a for a in run.get("artifacts", []) if a["kind"] == kind), None)
        if not artifact:
            continue
        data = download_artifact(run["run_id"], artifact["path"])
        if data:
            column.download_button(label, data, file_name=filename)


def render_chat(run: dict) -> None:
    """Follow-up questions, answered from the persisted run and the real model."""
    st.subheader("Ask about this model")
    st.caption(
        "Answered from the run's record and the fitted model file — "
        "not from a fresh guess."
    )

    key = f"chat_{run['run_id']}"
    st.session_state.setdefault(key, [])

    for role, text in st.session_state[key]:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("e.g. which feature mattered most?")
    if not question:
        return

    st.session_state[key].append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"), st.spinner("Checking the run record…"):
        try:
            response = requests.post(
                f"{API}/runs/{run['run_id']}/ask",
                json={"question": question},
                timeout=120,
            )
            response.raise_for_status()
            answer = response.json()["answer"]
        except requests.RequestException as exc:
            answer = f"Could not reach the API: {exc}"
        st.write(answer)
    st.session_state[key].append(("assistant", answer))


def render_log(run: dict) -> None:
    with st.expander("Agent log"):
        for message in run.get("messages", []):
            icon = {"info": "·", "warning": "⚠️", "error": "❌"}.get(message["level"], "·")
            st.text(f"{icon} [{message['agent']}] {message['content']}")
    usage = run.get("token_usage") or {}
    if usage:
        total = sum(u["cost_usd"] for u in usage.values())
        st.caption(f"LLM spend this run: ${total:.3f}")


# --- Page ---

st.title("🧪 Autonomous Data Scientist")
st.caption("Upload a CSV, state a goal in plain English, get a model and a deck.")

with st.sidebar:
    st.header("New run")
    uploaded = st.file_uploader("CSV", type=["csv"])
    goal = st.text_input("Goal", placeholder="predict churn")
    if st.button("Run", type="primary", disabled=not (uploaded and goal.strip())):
        try:
            response = requests.post(
                f"{API}/runs",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                data={"goal": goal},
                timeout=60,
            )
            response.raise_for_status()
            st.session_state["run_id"] = response.json()["run_id"]
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not start the run: {exc}")

    st.divider()
    st.header("Past runs")
    runs = api_get("/runs") or []
    for row in runs[:15]:
        if st.button(
            f"{row['status']} · {row['user_goal'][:28]}",
            key=row["run_id"],
            use_container_width=True,
        ):
            st.session_state["run_id"] = row["run_id"]
            st.rerun()

    health = api_get("/health")
    st.caption(f"API: {health['status']}" if health else "API: unreachable")

run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Upload a CSV and give it a goal to get started. "
            "Try `sample_data/customer_churn.csv` with “predict churn”.")
    st.stop()

run = api_get(f"/runs/{run_id}")
if not run:
    st.stop()

st.subheader(run["user_goal"])
left, mid, right = st.columns(3)
left.metric("Status", run["status"])
mid.metric("Task", str(run.get("task_type") or "—"))
right.metric("Confidence", f"{(run.get('confidence') or 0):.0%}")

render_progress(run)

if run.get("plan"):
    with st.expander("The agent's plan"):
        for i, step in enumerate(run["plan"], 1):
            st.markdown(f"{i}. {step}")

if run["status"] == "awaiting_human":
    render_escalation(run)
elif run["status"] == "failed":
    st.error(f"This run ended without a result: {run.get('error')}")
elif run["status"] == "completed":
    render_results(run)
    render_chat(run)

render_log(run)

if run["status"] == "running":
    time.sleep(POLL_SECONDS)
    st.rerun()
