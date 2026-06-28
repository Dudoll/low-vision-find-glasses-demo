# Phase 2: Object Vocabulary and Intent Parser

## Goal

Implement target-object parsing and alias mapping.

Do not implement ASR in this phase. The parser receives plain text.

## Files to create or modify

- `lvfind/intent/object_vocab.py`
- `lvfind/intent/parser.py`
- `configs/objects_zh.yaml`
- `tests/test_object_vocab.py`
- `tests/test_intent_parser.py`

## Requirements

### Object vocabulary

Support canonical object names and aliases.

Example:

```yaml
objects:
  手机:
    aliases: ["手机", "电话", "cell phone", "phone"]
    detector_labels: ["cell phone"]
    typical_width_m: 0.075
```

The vocabulary module should support:

- Load from YAML.
- Find canonical object by alias.
- Return detector labels for canonical object.
- Return unknown object cleanly if no match exists.

### Intent parser

Support Chinese commands such as:

- 帮我找手机
- 找一下杯子
- 我要找遥控器
- 手机在哪里
- 帮我看看杯子在哪里

Return a structured result:

```python
{
    "intent": "find_object",
    "target": "手机",
    "raw_target": "手机"
}
```

For unsupported text, return an explicit no-match result.

## Tests

Add at least 10 Chinese command tests.

Run:

```bash
python -m pytest tests/test_object_vocab.py tests/test_intent_parser.py
```

## Definition of done

- Parser is independent from ASR.
- Object labels are not hardcoded in detector code.
- Unknown object behavior is tested.
