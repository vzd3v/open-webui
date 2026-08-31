from __future__ import annotations

import logging
from typing import Any

from fastapi.responses import JSONResponse
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import get_last_user_message, get_message_list
from open_webui.utils.payload import apply_params_to_form_data
from open_webui.utils.task import (
    prompt_template,
    prompt_variables_template,
    replace_messages_variable,
    replace_prompt_variable,
)

log = logging.getLogger(__name__)

DEFAULT_CONTEXT_COMPACTION_PROMPT = """### Task:
Summarize the conversation history that will be compacted out of the active chat context.

### Instructions:
- Preserve key decisions, user preferences, and constraints.
- Preserve files, artifacts, tool results, and code changes that matter going forward.
- Preserve the current task state, unresolved questions, and next steps.
- Be factual and specific. Do not invent details.
- Keep the summary concise, but complete enough for the assistant to continue without the removed messages.

### Previous Summary:
{{PREVIOUS_SUMMARY}}

### Messages Being Compacted:
{{COMPACTED_MESSAGES}}

### Recent Messages Kept In Context:
{{RECENT_MESSAGES}}"""

CONTEXT_SUMMARY_START = '<openwebui_context_summary>'
CONTEXT_SUMMARY_END = '</openwebui_context_summary>'
CONTEXT_SUMMARY_OUTPUT_INSTRUCTION = (
    f'End your response with {CONTEXT_SUMMARY_END}. Begin your response with {CONTEXT_SUMMARY_START}.\n'
    'Place only the final summary between these markers. Do not call tools or include analysis.'
)


async def compact_messages_for_request(
    request,
    user,
    messages: list[dict],
    metadata: dict,
    model_id: str,
    models: dict,
    system_prompt: str = '',
) -> tuple[list[dict], str | None, bool]:
    config = await _load_config()
    if not config['enable']:
        return messages, None, False

    system_messages = [messages[0]] if messages and messages[0].get('role') == 'system' else []
    messages = messages[1:] if system_messages else messages

    messages, previous_summary = _apply_latest_summary_checkpoint(messages)
    token_threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata)
    if not _exceeds_token_threshold(messages, system_prompt, previous_summary, token_threshold) or len(messages) <= 3:
        return [*system_messages, *messages], previous_summary, False

    boundary = _find_compaction_boundary(messages, config['retention_percentage'])
    compacted_messages = messages[:boundary]
    recent_messages = messages[boundary:]
    if not compacted_messages or not recent_messages:
        return [*system_messages, *messages], previous_summary, False

    event_emitter = None
    if metadata.get('chat_id') and metadata.get('message_id'):
        from open_webui.socket.main import get_event_emitter

        event_emitter = await get_event_emitter(metadata)

    if event_emitter:
        await event_emitter(
            {
                'type': 'context_compaction',
                'data': {
                    'action': 'context_compaction',
                    'description': 'Compacting context',
                    'done': False,
                },
            }
        )

    try:
        summary = await _generate_summary(
            request,
            user,
            model_id,
            models,
            compacted_messages,
            recent_messages,
            previous_summary,
            config['prompt_template'],
        )
    except Exception:
        if event_emitter:
            await event_emitter(
                {
                    'type': 'context_compaction',
                    'data': {
                        'action': 'context_compaction',
                        'description': 'Context compaction failed',
                        'done': True,
                        'error': True,
                    },
                }
            )
        raise

    chat_id = metadata.get('chat_id')
    checkpoint_message_id = (
        recent_messages[0].get('id') or metadata.get('user_message_id') or metadata.get('message_id')
    )
    if is_saved_chat_id(chat_id) and checkpoint_message_id:
        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            checkpoint_message_id,
            {'contextSummary': summary},
            touch=False,
        )

    log.info(
        'Compacted chat context for chat=%s checkpoint=%s response=%s dropped=%d kept=%d summary_chars=%d',
        chat_id,
        checkpoint_message_id,
        metadata.get('message_id'),
        len(compacted_messages),
        len(recent_messages),
        len(summary),
    )

    if event_emitter:
        await event_emitter(
            {
                'type': 'context_compaction',
                'data': {
                    'action': 'context_compaction',
                    'description': 'Context compacted',
                    'done': True,
                },
            }
        )

    return [*system_messages, *recent_messages], summary, True


async def compact_chat_branch(request, user, chat: Any, model_id: str, models: dict) -> dict:
    config = await _load_config()
    if not config['enable']:
        return {'ok': True, 'compacted': False, 'reason': 'disabled'}

    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return {'ok': True, 'compacted': False, 'reason': 'empty'}

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    if not messages_map:
        messages_map = history.get('messages') or {}

    messages, previous_summary = _apply_latest_summary_checkpoint(get_message_list(messages_map, current_id))
    compacted_messages = messages[:-1]
    recent_messages = messages[-1:]
    if not compacted_messages or not recent_messages:
        return {'ok': True, 'compacted': False, 'reason': 'too_short'}

    summary = await _generate_summary(
        request,
        user,
        model_id,
        models,
        compacted_messages,
        recent_messages,
        previous_summary,
        config['prompt_template'],
    )
    await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat.id, current_id, {'contextSummary': summary}, touch=False
    )

    return {
        'ok': True,
        'compacted': True,
        'dropped_messages': len(compacted_messages),
        'kept_messages': len(recent_messages),
        'summary_chars': len(summary),
    }


async def _load_config() -> dict:
    values = await Config.get_many(
        'chat.context_compaction.enable',
        'chat.context_compaction.token_threshold',
        'chat.context_compaction.token_cap',
        'chat.context_compaction.retention_percentage',
        'chat.context_compaction.prompt_template',
    )
    token_threshold = _parse_positive_int(values.get('chat.context_compaction.token_threshold')) or 80000
    return {
        'enable': bool(values.get('chat.context_compaction.enable', False)),
        'token_threshold': token_threshold,
        'token_cap': _parse_positive_int(values.get('chat.context_compaction.token_cap')) or token_threshold,
        'retention_percentage': _clamp_retention_percentage(values.get('chat.context_compaction.retention_percentage')),
        'prompt_template': values.get('chat.context_compaction.prompt_template', '') or '',
    }


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _clamp_retention_percentage(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 40
    return min(50, max(10, parsed))


def _resolve_token_threshold(global_threshold: int, global_cap: int, metadata: dict) -> int:
    configured_threshold = _parse_positive_int((metadata.get('params') or {}).get('compact_token_threshold'))
    return min(configured_threshold or global_threshold, global_cap)


def _usage_token_count(usage: dict) -> int:
    prompt_tokens = int(usage.get('prompt_tokens') or usage.get('prompt_eval_count') or 0)
    if not prompt_tokens and (usage.get('prompt_n') is not None or usage.get('cache_n') is not None):
        prompt_tokens = int(usage.get('prompt_n') or 0) + int(usage.get('cache_n') or 0)
    if not prompt_tokens:
        prompt_tokens = int(usage.get('input_tokens') or 0)

    completion_tokens = int(
        usage.get('completion_tokens')
        or usage.get('output_tokens')
        or usage.get('eval_count')
        or usage.get('predicted_n')
        or 0
    )
    return prompt_tokens + completion_tokens


async def get_chat_context_usage(chat: Any, model_id: str | None = None) -> dict | None:
    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return None

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    messages = get_message_list(messages_map or history.get('messages') or {}, current_id)
    if not messages:
        return None

    config = await _load_config()
    if not config['enable']:
        return None

    params = ((chat.chat or {}).get('params') or {}).copy()
    if model_id:
        params['model'] = model_id
    threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], {'params': params})
    messages, previous_summary = _apply_latest_summary_checkpoint(messages)

    for idx in range(len(messages) - 1, -1, -1):
        usage = messages[idx].get('usage') or (messages[idx].get('info') or {}).get('usage')
        if isinstance(usage, dict) and (tokens := _usage_token_count(usage)):
            tokens += _estimate_messages_tokens(messages[idx + 1 :])
            return _build_context_usage(tokens, threshold)

    tokens = _estimate_tokens(previous_summary or '') + _estimate_messages_tokens(messages)
    return _build_context_usage(tokens, threshold)


def _build_context_usage(tokens: int, threshold: int) -> dict:
    return {
        'tokens': tokens,
        'estimated_tokens': tokens,
        'threshold': threshold,
        'percent': round((tokens / threshold) * 100) if threshold > 0 else 0,
        'source': 'estimated',
    }


def _apply_latest_summary_checkpoint(messages: list[dict]) -> tuple[list[dict], str | None]:
    summary = None
    summary_idx = None

    for idx, message in enumerate(messages):
        value = message.get('contextSummary') or message.get('context_summary')
        if isinstance(value, str) and value.strip():
            summary = value
            summary_idx = idx

    if summary_idx is None:
        return messages, None
    return messages[summary_idx:], summary


def _exceeds_token_threshold(messages: list[dict], system_prompt: str, summary: str | None, threshold: int) -> bool:
    if threshold <= 0:
        return False

    for idx in range(len(messages) - 1, -1, -1):
        usage = messages[idx].get('usage') or (messages[idx].get('info') or {}).get('usage')
        if isinstance(usage, dict) and (tokens := _usage_token_count(usage)):
            return tokens + _estimate_messages_tokens(messages[idx + 1 :]) > threshold

    estimated = _estimate_tokens(system_prompt) + _estimate_tokens(summary or '') + _estimate_messages_tokens(messages)
    return estimated > threshold


def _find_compaction_boundary(messages: list[dict], retention_percentage: int = 40) -> int:
    retention_percentage = _clamp_retention_percentage(retention_percentage)
    keep_count = max(2, len(messages) * retention_percentage // 100)
    target = max(1, len(messages) - keep_count)
    boundaries = [idx for idx, message in enumerate(messages) if message.get('role') == 'user'][1:]
    return next((idx for idx in reversed(boundaries) if idx <= target), 0)


async def _generate_summary(
    request,
    user,
    model_id: str,
    models: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
    previous_summary: str | None,
    summary_prompt_template: str,
) -> str:
    from open_webui.utils.chat import generate_chat_completion

    task_config = await Config.get_many(
        'task.model.params',
        'chat.context_compaction.model',
    )
    context_compaction_model = task_config.get('chat.context_compaction.model')
    task_model_id = context_compaction_model if context_compaction_model in models else model_id
    if task_model_id not in models:
        raise ValueError('No available model for context compaction')

    summary_prompt_template = summary_prompt_template.strip() or DEFAULT_CONTEXT_COMPACTION_PROMPT
    all_messages = [*compacted_messages, *recent_messages]
    prompt = replace_prompt_variable(summary_prompt_template, get_last_user_message(all_messages) or '')
    prompt = replace_messages_variable(prompt, all_messages)
    prompt = replace_messages_variable(prompt, compacted_messages, 'COMPACTED_MESSAGES')
    prompt = replace_messages_variable(prompt, recent_messages, 'RECENT_MESSAGES')
    prompt = prompt_variables_template(prompt, {'{{PREVIOUS_SUMMARY}}': previous_summary or ''})
    prompt = await prompt_template(prompt, user)
    prompt = f'{prompt.rstrip()}\n\n{CONTEXT_SUMMARY_OUTPUT_INSTRUCTION}'

    task_model_params = task_config.get('task.model.params') or {}
    if not isinstance(task_model_params, dict):
        task_model_params = {}
    task_model_params = {key: value for key, value in task_model_params.items() if value is not None and value != ''}
    task_model_params = task_model_params or {
        'max_tokens': models[task_model_id].get('info', {}).get('params', {}).get('max_tokens', 1000)
    }

    payload = {
        'model': task_model_id,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'metadata': {
            **(request.state.metadata if hasattr(request.state, 'metadata') else {}),
            'task': 'context_compaction',
        },
    }

    payload = apply_params_to_form_data(payload, models[task_model_id], task_model_params)
    _remove_incompatible_task_params(payload)
    response = await generate_chat_completion(
        request,
        form_data=payload,
        user=user,
        bypass_model_params=True,
        bypass_system_prompt=True,
    )
    summary = _extract_summary(_response_text(response))
    if not summary:
        raise ValueError('Context compaction returned no valid summary')
    return summary


def _response_text(response: Any) -> str:
    response = _response_dict(response)
    if not response:
        return ''

    if response.get('choices'):
        return _chat_completion_text(response)
    return _responses_text(response)


def _response_dict(response: Any) -> dict:
    if isinstance(response, list) and len(response) == 1:
        response = response[0]

    if isinstance(response, JSONResponse):
        if response.status_code >= 400:
            return {}
        try:
            response = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
        except Exception:
            return {}

    return response if isinstance(response, dict) else {}


def _chat_completion_text(response: dict) -> str:
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        return ''

    choice = choices[0]
    if not isinstance(choice, dict) or choice.get('finish_reason') not in {None, 'stop'}:
        return ''

    message = choice.get('message') or {}
    if (
        not isinstance(message, dict)
        or message.get('role') not in {None, 'assistant'}
        or message.get('tool_calls')
        or message.get('function_call')
    ):
        return ''

    content = message.get('content')
    return content if isinstance(content, str) else ''


def _responses_text(response: dict) -> str:
    if response.get('status') not in {None, 'completed'}:
        return ''

    parts = []
    for item in response.get('output') or []:
        text = _responses_output_item_text(item)
        if text is None:
            return ''
        if text:
            parts.append(text)
    return '\n'.join(parts)


def _responses_output_item_text(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None

    item_type = item.get('type')
    if item.get('tool_calls') or item.get('function_call'):
        return None
    if item_type in {'reasoning', 'compaction'}:
        return ''
    if item_type != 'message':
        return None
    if item.get('role') != 'assistant':
        return None

    parts = []
    for content in item.get('content') or []:
        if not isinstance(content, dict) or content.get('type') not in {'text', 'output_text'}:
            return None
        text = content.get('text') or content.get('content')
        if not isinstance(text, str):
            return None
        parts.append(text)
    return '\n'.join(parts)


def _remove_incompatible_task_params(payload: dict) -> None:
    incompatible_keys = (
        'format',
        'function_call',
        'functions',
        'parallel_tool_calls',
        'response_format',
        'tool_choice',
        'tools',
    )
    for key in incompatible_keys:
        payload.pop(key, None)

    options = payload.get('options')
    if isinstance(options, dict):
        for key in incompatible_keys:
            options.pop(key, None)


def _extract_summary(text: str) -> str:
    if text.count(CONTEXT_SUMMARY_START) != 1 or text.count(CONTEXT_SUMMARY_END) != 1:
        return ''

    start = text.find(CONTEXT_SUMMARY_START) + len(CONTEXT_SUMMARY_START)
    end = text.find(CONTEXT_SUMMARY_END, start)
    if end < start:
        return ''
    summary = text[start:end].strip()
    return '' if summary in {'...', '…'} else summary


def _estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += 4
        content = message.get('content')
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    total += _estimate_tokens(item)
                elif item.get('type') in {'image', 'image_url'}:
                    total += 1000
                else:
                    total += _estimate_tokens(item.get('text') or item.get('content') or item)
        else:
            total += _estimate_tokens(content)

        total += _estimate_tokens(message.get('output'))
        total += _estimate_tokens(message.get('tool_calls'))
        total += _estimate_tokens(message.get('files'))
    return total


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0

    if not isinstance(value, str):
        try:
            value = JSONCodec.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)

    if not value:
        return 0

    return max(1, len(value) // 4)
