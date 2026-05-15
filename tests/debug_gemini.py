"""Debug script — python -m tests.debug_gemini"""
import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference.course_finder import CourseFinder


async def main():
    f = CourseFinder()

    print("=== Test _find_free_resources (TypeScript) ===")
    try:
        result = await f._find_free_resources("TypeScript", "Frontend Engineer")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print("ERROR:", e)

    print()
    print("=== Test _find_free_resources (Kubernetes) ===")
    try:
        result = await f._find_free_resources("Kubernetes", "DevOps Engineer")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print("ERROR:", e)

    print()
    print("=== Test full find_courses_for_skills ===")
    try:
        result = await f.find_courses_for_skills(
            ["TypeScript", "Docker", "Kubernetes"], "Frontend Engineer"
        )
        for skill, platforms in result.items():
            print(f"\n--- {skill} ---")
            for platform, items in platforms.items():
                print(f"  {platform}:")
                for c in items:
                    st = f" [{c.get('source_type')}]" if c.get("source_type") else ""
                    print(f"    - {c.get('title', '?')}{st}")
                    print(f"      {c.get('url', '?')}")
    except Exception as e:
        print("ERROR:", e)

    print()
    print("=== Test roadmap.sh URL validation ===")
    slugs = ["typescript", "kubernetes", "docker", "react", "zustand-xyz-fake"]
    for slug in slugs:
        url = f"https://roadmap.sh/{slug}"
        ok = await f._url_exists_get(url)
        print(f"  [{'OK' if ok else 'FAIL'}] {url}")


asyncio.run(main())
