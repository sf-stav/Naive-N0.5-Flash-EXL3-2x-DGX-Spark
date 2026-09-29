"""Grammar-constrained Qwen3-Coder tool calling for Naive-N0.5-Flash.

The chat template asks for Qwen3-Coder XML tool calls::

    <tool_call>
    <function=NAME>
    <parameter=KEY>VALUE</parameter>
    </function>
    </tool_call>

At 3.5 bpw the model reliably emits the ``<tool_call>`` token when it wants a tool, but it
often continues in a loose form such as ``<tool_call> function=getWeather city="Lisbon" ...``.
It also sometimes closes a parameter as ``</parameter days>`` or ``</ value``. No parser
accepts these.

vLLM's built-in ``qwen_3_coder`` structural tag does not catch these:
- it only triggers on the full ``<tool_call>\\n<function=`` prefix;
- for ``tool_choice="auto"`` it is only built when a tool is marked ``strict``;
- its XML string values accept anything up to an exact ``</parameter>``, so a garbled close
  ends up inside the value.

This parser keeps vLLM's Qwen3 parsing (and argument type coercion) unchanged and supplies its
own grammar:
- it triggers on ``<tool_call>`` alone;
- it forces ``\\n<function=`` plus one of the declared tool names;
- then one ``<parameter=KEY>VALUE</parameter>`` per schema property, in schema order, with
  optional properties skippable.

Values are constrained by type:
- string: free text; closing tags such as ``</div>`` are allowed after the first line;
- integer / number: a numeral;
- boolean: ``true`` / ``false``;
- enum: one of its values;
- object / array: JSON.

Text before the call, including the ``<think>`` block, is unconstrained.

Load with ``--tool-parser-plugin <this file> --tool-call-parser naive_n05``.
"""

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

TRIGGER = "<tool_call>"
_INT = r"\s*-?[0-9]+\s*"
_NUM = r"\s*-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?\s*"
# String values. The model's malformed parameter closes ("hello.c</ hello.c", "hello.c</hello.c",
# "Lisbon</parameter days>", ...) always come right after a short value, on its first line. So:
# - first line: free text in which "<" is always followed by a character ("a<b", "<<",
#   "#include <x>" are fine) but "</" never appears, so a one-line value can only end with the
#   real "</parameter>";
# - after the first newline (file contents, code, HTML): "</" may start a real closing tag
#   ("</div>"), but never "</param...", "</function", "</tool_call" or "</" + non-letter.
_AFTER_CLOSE = (
    "([a-eg-oq-su-zA-Z]|p[^a<]|pa[^r<]|par[^a<]|para[^m<]"
    "|f[^u<]|fu[^n<]|fun[^c<]|func[^t<]|funct[^i<]|functi[^o<]|functio[^n<]"
    "|t[^o<]|to[^o<]|too[^l<]|tool[^_<])"
)
_STRING = (
    r"([^<\n]|<+[^/<\n])*"
    r"(\n([^<]|<+[^/<]|</" + _AFTER_CLOSE + r")*)?"
)


def _value_format(schema):
    from xgrammar.structural_tag import (
        ConstStringFormat,
        JSONSchemaFormat,
        OrFormat,
        RegexFormat,
    )

    schema = schema if isinstance(schema, dict) else {}
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        values = [v if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
                  for v in schema["enum"]]
        return OrFormat(elements=[ConstStringFormat(value=v) for v in values])
    kind = schema.get("type")
    if kind == "integer":
        return RegexFormat(pattern=_INT)
    if kind == "number":
        return RegexFormat(pattern=_NUM)
    if kind == "boolean":
        return OrFormat(elements=[ConstStringFormat(value="true"), ConstStringFormat(value="false")])
    if kind in ("object", "array"):
        return JSONSchemaFormat(json_schema=schema, style="json")
    # string, missing or union types: free text, see _STRING
    return RegexFormat(pattern=_STRING)


def _function_tag(fn):
    from xgrammar.structural_tag import (
        ConstStringFormat,
        OptionalFormat,
        SequenceFormat,
        TagFormat,
    )

    params = fn.parameters if isinstance(fn.parameters, dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = set(params.get("required") or [])
    elements = []
    for key, sub in props.items():
        tag = TagFormat(begin=f"<parameter={key}>", content=_value_format(sub),
                        end="</parameter>\n")
        elements.append(tag if key in required else OptionalFormat(content=tag))
    if not elements:
        content = ConstStringFormat(value="")
    elif len(elements) == 1:
        content = elements[0]
    else:
        content = SequenceFormat(elements=elements)
    return TagFormat(begin=f"<tool_call>\n<function={fn.name}>\n", content=content,
                     end="</function>\n</tool_call>")


def _named_choice(tool_choice):
    fn = getattr(tool_choice, "function", None)
    name = getattr(fn, "name", None) or getattr(tool_choice, "name", None)
    return name if isinstance(name, str) else None


@ToolParserManager.register_module("naive_n05")
class NaiveN05ToolParser(Qwen3EngineToolParser):
    structural_tag_model = "naive_n05"  # non-None: vLLM asks get_structural_tag()

    def get_structural_tag(self, request, *, reasoning: bool = False):
        from xgrammar import StructuralTag
        from xgrammar.structural_tag import TriggeredTagsFormat

        choice = request.tool_choice
        if choice == "none" or not request.tools:
            return None
        functions = [t.function for t in request.tools
                     if getattr(getattr(t, "function", None), "name", None)]
        named = None if isinstance(choice, str) else _named_choice(choice)
        if named is not None:
            functions = [f for f in functions if f.name == named]
        if not functions:
            return None
        parallel = getattr(request, "parallel_tool_calls", True)
        return StructuralTag(
            format=TriggeredTagsFormat(
                triggers=[TRIGGER],
                tags=[_function_tag(fn) for fn in functions],
                at_least_one=choice == "required" or named is not None,
                stop_after_first=parallel is False,
            )
        )
