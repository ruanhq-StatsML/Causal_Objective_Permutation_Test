# Federated FSDS — Legal / Finance / Product agent blocks

Multi-agent sales copilot: Legal reviews clauses, Finance checks margin, Product personalizes. Federated FSDS detects regulatory drift on legal block only; finance/product stay stable — OFS targets legal features, not global model shutdown.

## Run

```bash
cd Python && python3 demo_federated_legal_agent_blocks.py
```

Artifact: `artifacts/federated_legal_agent_blocks.json`.

## Business impact

- Top drift block: `legal_agent_block`
- Simulated revenue unlock (quarter): **$125,000** when monitor enables controlled rollout vs hard stop
- Each block ships an **impact receipt** for audit (`impact_receipts[]`).
