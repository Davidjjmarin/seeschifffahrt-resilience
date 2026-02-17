# German Seaport Resilience & Concentration (Destatis) — Python

A compact, reproducible data analysis project that benchmarks **German seaport regions** by how **diversified vs. dependent** their activity mix is.

Using the official **Destatis “Seeschifffahrt” (sea shipping) XLSX** tables, the project builds a **mix-based resilience view** across three activity streams:
- **Cargo** (tons)
- **Containers** (TEU)
- **Passengers** (count)

It then quantifies:
- **Concentration / dependency** using **HHI (Herfindahl–Hirschman Index)**  
- A simple **Resilience Score (0–100)** (mostly diversification + small YoY momentum component)

The output is an “Economist-style” **one-pager graphic** designed for fast interpretation and sharing.

---

## Outputs

- `output/linkedin_seaport_resilience_onepager.png` (main one-pager)
- `output/bundesland_resilience_metrics.csv` (HHI, shares, scores)
- `output/bundesland_activity_long.csv` (cleaned long format data)
