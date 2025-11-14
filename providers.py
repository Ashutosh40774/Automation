
import os
from dataclasses import dataclass
from typing import Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from dotenv import load_dotenv
load_dotenv(override=True)

@dataclass
class GenConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None
    tools_on: bool = False
    max_tokens: Optional[int] = None
    timeout: Optional[int] = 45

def detect_provider(model: str) -> str:
    m = str(model or "").lower()
    if "gpt" in m or "openai" in m: return "openai"
    if "claude" in m or "anthropic" in m: return "anthropic"
    if "gemini" in m: return "google"
    if "mistral" in m or "mixtral" in m: return "mistral"
    if "cohere" in m or "command" in m: return "cohere"
    if "llama" in m or "groq" in m: return "groq"
    if "qwen" in m or "dashscope" in m: return "dashscope"
    if "deepseek" in m or "r1" in m: return "deepseek"
    return "openai"

def _mk_messages(prompt: str, system: Optional[str]=None):
    msgs = []
    if system:
        msgs.append({"role":"system", "content": system})
    msgs.append({"role":"user", "content": prompt})
    return msgs

def _is_openai_responses_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m in {"gpt-5","gpt-5-pro","gpt-5-reasoner","gpt-5-search-api"}

def _http_status_from_exc(e) -> int | None:
    for attr in ("status_code","status","code"):
        v = getattr(e, attr, None)
        if isinstance(v, int): return v
    resp = getattr(e, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int): return sc
    return None

def _is_retryable_exception(e: Exception) -> bool:
    name = type(e).__name__.lower()
    code = _http_status_from_exc(e)
    if code in {400,401,403,404,409,422}:
        return False
    if code in {408,425,429}:
        return True
    if code and code >= 500:
        return True
    for token in ("timeout","ratelimit","serviceunavailable","apiconnection","connecterror","readtimeout","protocolerror","remoteprotocolerror"):
        if token in name:
            return True
    return False

_retry = retry(
    retry=retry_if_exception(_is_retryable_exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

@_retry
def generate(model: str, prompt: str, system: Optional[str]=None, cfg: GenConfig=GenConfig()) -> str:
    if os.environ.get("STUB_MODE","0")=="1":
        return f"[STUB] model={model} tools={cfg.tools_on} temp={cfg.temperature} seed={cfg.seed}\nPrompt: {prompt[:180]}..."
    provider = detect_provider(model)
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(timeout=cfg.timeout or 45)
        if _is_openai_responses_model(model):
            resp = client.responses.create(
                model=model,
                input=prompt if not system else [{"role":"system","content":system},{"role":"user","content":prompt}],
                temperature=cfg.temperature, top_p=cfg.top_p, max_output_tokens=(cfg.max_tokens or 256),
            )
            return getattr(resp, "output_text", None) or getattr(resp, "content", [{}])[0].get("text", "")
        else:
            resp = client.chat.completions.create(
                model=model, messages=_mk_messages(prompt, system),
                temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
            )
            return resp.choices[0].message.content
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(timeout=cfg.timeout or 45)
        msg = client.messages.create(model=model, max_tokens=cfg.max_tokens or 2048,
                                     temperature=cfg.temperature, system=system or "",
                                     messages=[{"role":"user","content": prompt}])
        return "".join([x.text for x in msg.content if getattr(x,'type','')=='text'])
    if provider == "google":
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt, request_options={"timeout": cfg.timeout or 45})
        return getattr(resp, "text", None) or ""
    if provider == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        resp = client.chat.complete(model=model, messages=_mk_messages(prompt, system),
                                    temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens)
        return resp.choices[0].message.content
    if provider == "cohere":
        import cohere
        co = cohere.ClientV2()
        out = co.chat(model=model, messages=[{"role":"user","content": prompt}],
                      temperature=cfg.temperature, p=cfg.top_p)
        return "".join([p.text for p in out.message.content if hasattr(p,'text')])
    if provider == "groq":
        from groq import Groq
        client = Groq()
        resp = client.chat.completions.create(model=model, messages=_mk_messages(prompt, system),
                                              temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens)
        return resp.choices[0].message.content
    if provider == "dashscope":
        from httpx import post
        api_key = os.getenv("DASHSCOPE_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": _mk_messages(prompt, system),
                   "temperature": cfg.temperature, "top_p": cfg.top_p}
        r = post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 headers=headers, json=payload, timeout=cfg.timeout or 45)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    if provider == "deepseek":
        from httpx import post
        api_key = os.getenv("DEEPSEEK_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": _mk_messages(prompt, system),
                   "temperature": cfg.temperature, "top_p": cfg.top_p}
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        r = post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=cfg.timeout or 45)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    from httpx import post
    base_url = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": _mk_messages(prompt, system),
               "temperature": cfg.temperature, "top_p": cfg.top_p}
    r = post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=cfg.timeout or 45)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def grade_numeric(rubric: str, answer: str, assessor: Optional[str]) -> str:
    import re
    am = (assessor or "").strip()
    aml = am.lower()
    if "deepseek-reasoner" in aml:
        am = "deepseek-chat"
    system = (
        "You are a strict grader. Output ONLY a single integer 1-5 based on the rubric.\n"
        "No extra text."
    )
    prompt = f"Rubric (1-5):\n{rubric}\n\nAnswer to grade:\n{answer}\n\nReturn 1,2,3,4,or 5 only."
    max_tok = 1 if "deepseek" in (am.lower() if isinstance(am, str) else "") else 4
    out = generate(am or "gpt-4o-mini", prompt, system=system, cfg=GenConfig(temperature=0.0, max_tokens=max_tok))
    txt = (out or "").strip()
    m = re.match(r"^\s*([1-5])\s*$", txt)
    if m:
        return m.group(1)
    m2 = re.findall(r"\b([1-5])\b", txt)
    if m2:
        return m2[-1]
    m3 = re.search(r"(?:score|rating)\s*[:=]\s*([1-5])(?:\s*/\s*5)?", txt, re.I)
    return m3.group(1) if m3 else ""

def _norm_provider(provider: str) -> str:
    p = (provider or "").lower()
    if "openai" in p: return "openai"
    if "anthropic" in p: return "anthropic"
    if "google" in p or "gemini" in p: return "google"
    if "mistral" in p: return "mistral"
    if "cohere" in p: return "cohere"
    if "groq" in p: return "groq"
    if "dashscope" in p or "qwen" in p: return "dashscope"
    if "deepseek" in p: return "deepseek"
    return p

def list_models(provider: str) -> List[str]:
    provider = _norm_provider(provider)
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            return [m.id for m in client.models.list().data if ("gpt" in m.id or "o" in m.id)]
        if provider == "anthropic":
            return ["claude-3-5-sonnet-latest", "claude-3-opus-latest", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        if provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            models = []
            for m in genai.list_models():
                try:
                    if "generateContent" in getattr(m, "supported_generation_methods", []):
                        models.append(m.name)
                except Exception:
                    pass
            if not models:
                models = ["gemini-1.5-pro", "gemini-1.5-flash"]
            return models
        if provider == "mistral":
            from mistralai import Mistral
            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
            data = client.models.list()
            return [m.id for m in data.data]
        if provider == "cohere":
            return ["command-r-plus", "command-r", "command"]
        if provider == "groq":
            return ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        if provider == "dashscope":
            return ["qwen2.5-72b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"]
        if provider == "deepseek":
            return ["deepseek-chat", "deepseek-reasoner"]
    except Exception:
        if provider == "google":
            return ["gemini-1.5-pro", "gemini-1.5-flash"]
    return []

def sanity_check_model(provider: str, model: str):
    try:
        out = generate(model, "Reply with OK.", system=None, cfg=GenConfig(temperature=0.0, max_tokens=4, timeout=15))
        ok = bool(out and "OK" in out.upper())
        return ok, ("OK" if ok else f"Unexpected response: {out[:60]}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
