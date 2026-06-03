import os
from pathlib import Path

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(override=True)

    for d in ["uploads", "vectorstore", "static", "templates", "data"]:
        Path(d).mkdir(exist_ok=True)

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        print("\n⚠️  GROQ_API_KEY not set in .env file!")
        print("   Get free key: https://console.groq.com\n")
    else:
        print(f"✅ Groq API key loaded ({key[:8]}...)")

    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Company GPT v2 starting...")
    print(f"   ➜  http://localhost:{port}")
    print(f"\n👤 Default login accounts:")
    print(f"   admin  / admin123  (Full access)")
    print(f"   editor / editor123 (Upload + Query)")
    print(f"   viewer / viewer123 (Query only)")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
