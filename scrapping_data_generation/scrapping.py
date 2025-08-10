# import asyncio, re
# from urllib.parse import urljoin
# from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# URL = "https://foorilla.com/hiring"

# async def run():
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(
#             headless=True,
#             args=["--disable-blink-features=AutomationControlled"]
#         )
#         context = await browser.new_context(
#             viewport={"width": 1400, "height": 900},
#             user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                         "AppleWebKit/537.36 (KHTML, like Gecko) "
#                         "Chrome/124.0.0.0 Safari/537.36"),
#             locale="en-US",
#         )
#         page = await context.new_page()
#         page.set_default_timeout(45000)

#         # 1) Load page
#         await page.goto(URL, wait_until="networkidle")

#         # 2) Open the Topics offcanvas
#         topics_btn = page.locator(
#             'a.fw-semibold[role="button"][data-bs-toggle="offcanvas"][data-bs-target="#oc_0"][hx-get="/topics/hiring/"]'
#         )
#         if await topics_btn.count() == 0:
#             topics_btn = page.locator(":is(a,button)[role='button']",
#                                       has_text=re.compile(r"Topics", re.I))
#         await topics_btn.first.wait_for(state="attached")
#         await topics_btn.first.click()

#         # 3) Wait until the offcanvas is open & content loaded
#         offcanvas = page.locator("#oc_0")
#         await offcanvas.wait_for(state="attached")
#         try:
#             await page.wait_for_selector("#oc_0.offcanvas.show", timeout=8000)
#         except PWTimeout:
#             await offcanvas.wait_for(state="visible", timeout=8000)

#         # Wait until HTMX replaced "..."
#         await page.wait_for_function(
#             """() => {
#                 const el = document.querySelector('#oc_0');
#                 if (!el) return false;
#                 const t = (el.textContent || '').trim();
#                 return t && t !== '...';
#             }""",
#             timeout=12000
#         )

#         # 4) Find "Artificial Intelligence" (robust matching)
#         patterns = [
#             r"^\s*Artificial\s+Intelligence\s*$",  # exact
#             r"Artificial\s*Intell",                # partial / typo-tolerant
#             r"\bAI\b",                             # short
#             r"Artificial"                          # last resort
#         ]
#         target = None
#         for pat in patterns:
#             cand = offcanvas.locator(
#                 ":is(.list-group-item, .dropdown-item, a, button)",
#                 has_text=re.compile(pat, re.I)
#             )
#             if await cand.count() > 0:
#                 target = cand.first
#                 break
#         if not target:
#             raise RuntimeError("Could not find a topic that looks like 'Artificial Intelligence'.")

#         # Sidebar job selectors
#         sidebar_col_selector = ".col.overflow-y-scroll.overflow-x-none.border-end.text-break"
#         sidebar_items_selector = f"{sidebar_col_selector} ul li.list-group-item"

#         # Count items BEFORE clicking the topic (to detect refresh)
#         try:
#             old_count = await page.locator(sidebar_items_selector).count()
#         except Exception:
#             old_count = -1

#         # Click with fallbacks
#         await target.scroll_into_view_if_needed()
#         try:
#             await target.click(timeout=5000)
#         except Exception:
#             try:
#                 await target.evaluate("el => el.click()")
#             except Exception:
#                 await target.focus()
#                 await offcanvas.press("Enter")

#         # 5) Wait for sidebar to refresh
#         #    Use a dict for args so we don't create 'selector,50' strings by accident.
#         try:
#             await page.wait_for_function(
#                 """(data) => {
#                     const el = document.querySelector(data.sel);
#                     if (!el) return false;
#                     const items = el.querySelectorAll('ul li.list-group-item');
#                     return items.length !== data.prev && items.length > 0;
#                 }""",
#                 arg={"sel": sidebar_col_selector, "prev": old_count},
#                 timeout=15000
#             )
#         except PWTimeout:
#             # Grace period in case the count is the same but contents changed
#             await page.wait_for_timeout(800)

#         # 6) Scrape sidebar jobs (title + URL if present)
#         job_items = page.locator(sidebar_items_selector)
#         count = await job_items.count()
#         jobs = []
#         for i in range(count):
#             li = job_items.nth(i)
#             title = (await li.inner_text()).strip()
#             a = li.locator("a").first
#             href = await a.get_attribute("href") if await a.count() > 0 else None
#             jobs.append({"title": title, "url": urljoin(URL, href) if href else None})

#         # Output (only jobs; no topics printed)
#         print(f"Jobs under 'Artificial Intelligence': {count}")
#         for j in jobs:
#             print("-", j["title"], "->", j["url"])

#         # 7) (Optional) Scrape current main table
#         table_rows = []
#         rows = page.locator(".container-fluid.g-0 .col-9.col-md-10 table tbody tr")
#         for tr in await rows.all():
#             tds = await tr.locator("td").all()
#             cells = [(await td.inner_text()).strip() for td in tds]
#             a = tr.locator("a").first
#             href = await a.get_attribute("href") if await a.count() > 0 else None
#             table_rows.append({"cells": cells, "url": urljoin(URL, href) if href else None})

#         if table_rows:
#             print("\nTable rows (current view):")
#             for r in table_rows:
#                 print("-", r["cells"], "->", r["url"])

#         await browser.close()

# if __name__ == "__main__":
#     asyncio.run(run())


# pip install playwright
# playwright install

# pip install playwright pandas tqdm
# playwright install

import asyncio, re
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from urllib.parse import urljoin
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

URL = "https://foorilla.com/hiring"

TOPIC_PATTERNS = {
    "Artificial Intelligence": [r"^\s*Artificial\s+Intelligence\s*$", r"Artificial\s*Intell", r"\bAI\b", r"Artificial"],
    "Big Data": [r"^\s*Big\s+Data\s*$", r"Big\s*Data"],
    "Computer Vision": [r"^\s*Computer\s+Vision\s*$", r"Computer\s*Vision", r"\bCV\b"],
    "Cloud Computing": [r"^\s*Cloud\s+Computing\s*$", r"Cloud\s*Computing", r"\bCloud\b"],
    "Data Science": [r"^\s*Data\s+Science\s*$", r"Data\s*Science"],
    "Machine Learning": [r"^\s*Machine\s+Learning\s*$", r"Machien\s+Learnig", r"Machine\s*Learn", r"\bML\b"],
    "MLOps": [r"^\s*MLOps\s*$", r"\bML\s*Ops\b", r"ML\s*Ops"],
    "Natural Language Processing": [r"^\s*Natural\s+Language\s+Processing\s*$", r"Natural\s+Language", r"\bNLP\b"],
}

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            locale="en-US",
        )
        page = await context.new_page()
        page.set_default_timeout(45000)

        topics_button_sel = 'a.fw-semibold[role="button"][data-bs-toggle="offcanvas"][data-bs-target="#oc_0"][hx-get="/topics/hiring/"]'
        offcanvas_sel = "#oc_0"
        sidebar_col_selector = ".col.overflow-y-scroll.overflow-x-none.border-end.text-break"
        sidebar_items_selector = f"{sidebar_col_selector} ul li.list-group-item"
        main_li_selector = ".container-fluid.g-0 .col-9.col-md-10 .row.g-0 ul.list-group.list-group-flush.text-break > li.list-group-item"
        detail_root = ".col.overflow-y-scroll.overflow-x-none.d-none.d-md-block.text-break"

        async def open_topics_offcanvas():
            btn = page.locator(topics_button_sel)
            if await btn.count() == 0:
                btn = page.locator(":is(a,button)[role='button']", has_text=re.compile(r"Topics", re.I))
            await btn.first.click()
            offcanvas = page.locator(offcanvas_sel)
            await offcanvas.wait_for(state="attached")
            try:
                await page.wait_for_selector(f"{offcanvas_sel}.offcanvas.show", timeout=8000)
            except PWTimeout:
                await offcanvas.wait_for(state="visible", timeout=8000)
            await page.wait_for_function(
                """() => {
                    const el = document.querySelector('#oc_0');
                    return el && el.textContent.trim() && el.textContent.trim() !== '...';
                }""",
                timeout=12000
            )
            return offcanvas

        async def select_topic(offcanvas, patterns):
            target = None
            for pat in patterns:
                cand = offcanvas.locator(
                    ":is(.list-group-item, .dropdown-item, a, button)",
                    has_text=re.compile(pat, re.I)
                )
                if await cand.count() > 0:
                    target = cand.first
                    break
            if not target:
                return False
            try:
                old_count = await page.locator(sidebar_items_selector).count()
            except Exception:
                old_count = -1
            try:
                await target.scroll_into_view_if_needed()
                await target.click(timeout=5000)
            except Exception:
                try:
                    await target.evaluate("el => el.click()")
                except Exception:
                    await target.focus()
                    await offcanvas.press("Enter")
            try:
                await page.wait_for_function(
                    """(data) => {
                        const el = document.querySelector(data.sel);
                        if (!el) return false;
                        const items = el.querySelectorAll('ul li.list-group-item');
                        return items.length !== data.prev && items.length > 0;
                    }""",
                    arg={"sel": sidebar_col_selector, "prev": old_count},
                    timeout=15000
                )
            except PWTimeout:
                await page.wait_for_timeout(800)
            return True

        async def safe_text(loc):
            try:
                if await loc.count() == 0:
                    return ""
                t = await loc.inner_text()
                return (t or "").strip()
            except Exception:
                return ""

        async def safe_texts(loc):
            out = []
            try:
                n = await loc.count()
                for i in range(n):
                    txt = await safe_text(loc.nth(i))
                    if txt:
                        out.append(txt)
            except Exception:
                pass
            return out

        async def scrape_jobs_for_current_topic(topic):
            try:
                await page.wait_for_selector(main_li_selector, timeout=8000)
            except PWTimeout:
                return []
            main_items = page.locator(main_li_selector)
            count = await main_items.count()
            rows = []
            for i in range(count):
                li = main_items.nth(i)
                before_sig = await page.evaluate(
                    """(sel) => {
                        const el = document.querySelector(sel);
                        if (!el) return '';
                        const len = (el.textContent || '').length;
                        const kids = el.querySelectorAll('*').length;
                        return `${len}:${kids}`;
                    }""",
                    detail_root
                )
                try:
                    await li.scroll_into_view_if_needed()
                    await li.click(timeout=5000)
                except Exception:
                    try:
                        await li.evaluate("el => el.click()")
                    except Exception:
                        await li.focus()
                        await page.keyboard.press("Enter")
                try:
                    await page.wait_for_function(
                        """(data) => {
                            const el = document.querySelector(data.sel);
                            if (!el) return false;
                            const len = (el.textContent || '').length;
                            const kids = el.querySelectorAll('*').length;
                            return `${len}:${kids}` !== data.prev;
                        }""",
                        arg={"sel": detail_root, "prev": before_sig},
                        timeout=15000
                    )
                except PWTimeout:
                    await page.wait_for_timeout(600)

                title_h1 = await safe_text(page.locator(f"{detail_root} .px-1.border-bottom h1"))
                hstacks = page.locator(f"{detail_root} .hstack.justify-content-between")
                hstack1 = await safe_text(hstacks.nth(0))
                hstack2 = await safe_text(hstacks.nth(1))
                ul_blocks = page.locator(f"{detail_root} .px-1.border-bottom.pb-2 ul.list-unstyled.mb-2")
                ul1_lis = await safe_texts(ul_blocks.nth(0).locator("li"))
                ul2_lis = await safe_texts(ul_blocks.nth(1).locator("li"))
                primary_emph = await safe_text(page.locator(f"{detail_root} .text-primary-emphasis.mb-2"))
                warning_wrap = await safe_text(page.locator(f"{detail_root} .text-warning-emphasis.text-wrap"))

                rows.append({
                    "topic": topic,
                    "job_title": title_h1,
                    "site": hstack1,
                    "basic info": hstack2,
                    "tasks": ul1_lis,
                    "Perks/Benefits": ul2_lis,
                    "Skills/Tech-stack required": primary_emph,
                    "Educational requirements": warning_wrap
                })
            return rows

        await page.goto(URL, wait_until="networkidle")
        all_rows = []

        # tqdm progress bar over topics
        for topic, patterns in tqdm_asyncio(TOPIC_PATTERNS.items(), desc="Processing topics", total=len(TOPIC_PATTERNS)):
            offcanvas = await open_topics_offcanvas()
            ok = await select_topic(offcanvas, patterns)
            if not ok:
                continue
            await page.wait_for_timeout(800)
            print(f"\nScraping topic: {topic}")
            rows = await scrape_jobs_for_current_topic(topic)
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        print(df)
        await browser.close()
        return df

if __name__ == "__main__":
    df = asyncio.run(run())
    df.to_csv("aijobs_jobs_scrapping.csv", index=False)


