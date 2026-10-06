"""Automation script to run live TabPFN-3.5 Data Generation demo via Web UI with Playwright."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
DEMO_DIR = Path("demos/05_data_generation")
CSV_PATH = DEMO_DIR / "synthetic_patients.csv"


def wait_for_server(url: str, timeout_sec: int = 20) -> bool:
    """Poll server until it answers on the given URL."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            res = httpx.get(f"{url}/api/session", timeout=2.0)
            if res.status_code in (200, 404):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        print(f"[!] Dataset missing at {CSV_PATH}. Run scripts/make_patient_synthetic.py first.")
        return 1

    print(f"[*] Target demo directory: {DEMO_DIR.resolve()}")

    # 1. Start uvicorn server in background
    print("[*] Starting local web server on port 8000...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tabchat.main:app", "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        print("[*] Waiting for server to become healthy...")
        if not wait_for_server(BASE_URL, timeout_sec=20):
            print("[!] Server failed to start within 20s.")
            out, err = server_proc.communicate(timeout=3)
            print("Server stdout:", out)
            print("Server stderr:", err)
            return 1
        print("[+] Server is online and responsive.")

        # 2. Launch Playwright
        print(f"[*] Launching Chrome via Playwright ({CHROME_PATH})...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
            )
            context = browser.new_context(viewport={"width": 1440, "height": 960})
            page = context.new_page()

            page.on("console", lambda msg: print(f"  [Browser Console {msg.type}] {msg.text}"))
            page.on("pageerror", lambda exc: print(f"  [Browser Error] {exc}"))

            print(f"[*] Navigating to {BASE_URL}...")
            page.goto(BASE_URL, wait_until="networkidle")

            # Wait for session initialization
            page.wait_for_function(
                "() => document.getElementById('session-id-display')?.textContent !== 'Initializing...'",
                timeout=10000,
            )
            session_id = page.locator("#session-id-display").text_content().strip()
            print(f"[+] Active Session ID: {session_id}")

            # 3. Upload Synthetic Patients dataset via file-input
            print(f"[*] Uploading Synthetic Patients dataset ({CSV_PATH})...")
            file_input = page.locator("#file-input")
            file_input.set_input_files(str(CSV_PATH.resolve()))

            # Wait for dataset card section to become visible
            page.wait_for_selector("#dataset-card-section:not(.hidden)", timeout=10000)
            shape_text = page.locator("#dataset-shape-badge").text_content().strip()
            print(f"[+] Dataset loaded successfully: {shape_text}")

            # 4. Human-like interaction: Type prompt in chat
            prompt = (
                "I want to validate this privacy-preserving synthetic clinical dataset by predicting patient treatment response (treatment_response) "
                "from biomarkers age, blood_pressure_sys, cholesterol, and glucose. Exclude patient_id as it is an ID, and evaluate with accuracy."
            )
            print(f"[*] Typing user prompt: '{prompt}'")
            chat_input = page.locator("#chat-input")
            chat_input.fill(prompt)

            print("[*] Submitting prompt to Gemini Planner...")
            page.click("#btn-send")

            # 5. Wait for Gemini Planner to respond and render the Plan Card
            print("[*] Waiting for Gemini Planner response and Plan Card (up to 60s)...")
            try:
                page.wait_for_selector("#btn-run-tabpfn", timeout=60000)
                print("[+] Plan Card rendered with confirmed JobSpec!")
            except Exception as e:
                print(f"[!] Plan Card did not appear: {e}")
                feed_text = page.locator("#chat-feed").text_content()
                print(f"[!] Current Chat Feed content:\n{feed_text}")
                raise

            # Small pause for UI rendering
            page.wait_for_timeout(2000)

            # 6. Click 'Confirm & Run TabPFN'
            print("[*] Clicking '🚀 Confirm & Run TabPFN' (calling live TabPFN-3.5 API)...")
            page.click("#btn-run-tabpfn")

            # 7. Wait for Execution and Narrator results card
            print("[*] Awaiting TabPFN-3.5 model fitting on synthetic data, baseline scoring, and Gemini narration (up to 120s)...")
            page.wait_for_selector("#active-results-card", timeout=120000)
            print("[+] Execution and narration finished! Results card rendered.")

            # Let SVG chart and text settle
            page.wait_for_timeout(3000)

            # 8. Save artifacts
            html_content = page.content()
            html_path = DEMO_DIR / "generation_session.html"
            html_path.write_text(html_content, encoding="utf-8")
            print(f"[+] Saved rendered HTML to: {html_path}")

            dash_png = DEMO_DIR / "generation_dashboard.png"
            page.screenshot(path=str(dash_png), full_page=True)
            print(f"[+] Saved full-page screenshot to: {dash_png}")

            results_card = page.locator("#active-results-card")
            if results_card.count() > 0:
                card_png = DEMO_DIR / "generation_results_card.png"
                results_card.screenshot(path=str(card_png))
                print(f"[+] Saved results card screenshot to: {card_png}")

            browser.close()

        # 9. Run CLI replay utility for terminal audit
        print(f"[*] Running CLI audit replay for session {session_id}...")
        replay_res = subprocess.run(
            [sys.executable, "-m", "tabchat.replay", session_id],
            capture_output=True,
            text=True,
        )
        audit_file = DEMO_DIR / "replay_audit.txt"
        audit_file.write_text(replay_res.stdout, encoding="utf-8")
        print(f"[+] Saved audit trail to: {audit_file}")
        print("\n--- AUDIT REPLAY SUMMARY ---")
        for line in replay_res.stdout.splitlines()[-35:]:
            print(line)

        print("\n[SUCCESS] Data Generation & Synthetic Validation demo completed successfully!")
        return 0

    finally:
        print("[*] Shutting down local web server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[*] Server stopped.")


if __name__ == "__main__":
    sys.exit(main())
