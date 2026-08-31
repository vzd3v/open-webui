import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _stub_module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture(scope='module')
def context_compaction_module():
    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.body = json.dumps(content).encode()
            self.status_code = status_code

    class JSONCodec:
        loads = staticmethod(json.loads)
        dumps = staticmethod(json.dumps)

    class Config:
        @staticmethod
        async def get_many(*_keys):
            return {}

    class Chats:
        pass

    async def prompt_template(prompt, _user):
        return prompt

    def get_last_user_message(messages):
        return next(
            (message.get('content', '') for message in reversed(messages) if message.get('role') == 'user'),
            '',
        )

    chat_module = _stub_module('open_webui.utils.chat', generate_chat_completion=AsyncMock())
    stubs = {
        'fastapi': _stub_module('fastapi'),
        'fastapi.responses': _stub_module('fastapi.responses', JSONResponse=JSONResponse),
        'open_webui': _stub_module('open_webui'),
        'open_webui.models': _stub_module('open_webui.models'),
        'open_webui.models.chats': _stub_module('open_webui.models.chats', Chats=Chats),
        'open_webui.models.config': _stub_module('open_webui.models.config', Config=Config),
        'open_webui.utils': _stub_module('open_webui.utils'),
        'open_webui.utils.chat': chat_module,
        'open_webui.utils.chat_id': _stub_module('open_webui.utils.chat_id', is_saved_chat_id=lambda _id: False),
        'open_webui.utils.json_codec': _stub_module('open_webui.utils.json_codec', JSONCodec=JSONCodec),
        'open_webui.utils.misc': _stub_module(
            'open_webui.utils.misc',
            get_last_user_message=get_last_user_message,
            get_message_list=lambda _messages, _current_id: [],
        ),
        'open_webui.utils.payload': _stub_module(
            'open_webui.utils.payload',
            apply_params_to_form_data=lambda payload, _model, _params: payload,
        ),
        'open_webui.utils.task': _stub_module(
            'open_webui.utils.task',
            prompt_template=prompt_template,
            prompt_variables_template=lambda prompt, _variables: prompt,
            replace_messages_variable=lambda prompt, _messages, *_args: prompt,
            replace_prompt_variable=lambda prompt, _value: prompt,
        ),
    }

    previous_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    module_path = Path(__file__).parents[1] / 'open_webui' / 'utils' / 'context_compaction.py'
    spec = importlib.util.spec_from_file_location('_context_compaction_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    yield module, chat_module

    sys.modules.pop('_context_compaction_under_test', None)
    for name, previous in previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _framed(module, summary='Keep the deployment constraint.'):
    return f'{module.CONTEXT_SUMMARY_START}\n{summary}\n{module.CONTEXT_SUMMARY_END}'


def test_chat_completions_accepts_framed_final_text(context_compaction_module):
    module, _ = context_compaction_module
    framed = _framed(module)
    response = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': framed}}]}

    assert module._extract_summary(module._response_text(response)) == 'Keep the deployment constraint.'


def test_json_response_wrapper_is_decoded(context_compaction_module):
    module, _ = context_compaction_module
    response = module.JSONResponse(
        content={'choices': [{'finish_reason': 'stop', 'message': {'content': _framed(module, 'Wrapped.')}}]}
    )

    assert module._extract_summary(module._response_text(response)) == 'Wrapped.'


def test_unframed_dsml_is_not_a_summary(context_compaction_module):
    module, _ = context_compaction_module
    dsml = (
        '<tool_protocol>function_calls><tool_protocol>invoke name="lookup_ticket">'
        '<tool_protocol>parameter name="id">12345'
    )
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': dsml}}]}

    assert module._response_text(response) == dsml
    assert module._extract_summary(module._response_text(response)) == ''


def test_json_response_http_error_is_rejected(context_compaction_module):
    module, _ = context_compaction_module
    response = module.JSONResponse(
        content={'choices': [{'finish_reason': 'stop', 'message': {'content': _framed(module, 'Do not use.')}}]},
        status_code=500,
    )

    assert module._response_text(response) == ''


def test_reasoning_only_chat_completion_is_rejected(context_compaction_module):
    module, _ = context_compaction_module
    response = {
        'choices': [
            {
                'finish_reason': 'stop',
                'message': {'content': None, 'reasoning_content': _framed(module, 'Internal reasoning.')},
            }
        ]
    }

    assert module._response_text(response) == ''


@pytest.mark.parametrize(
    'choice',
    [
        {'finish_reason': 'length', 'message': {'content': 'framed'}},
        {'finish_reason': 'tool_calls', 'message': {'content': 'framed'}},
        {'finish_reason': 'stop', 'message': {'content': 'framed', 'tool_calls': [{'id': 'call-1'}]}},
        {'finish_reason': 'stop', 'message': {'content': 'framed', 'function_call': {'name': 'lookup'}}},
    ],
    ids=['length', 'tool-finish', 'tool-calls', 'legacy-function-call'],
)
def test_non_final_chat_completions_are_rejected(context_compaction_module, choice):
    module, _ = context_compaction_module

    assert module._response_text({'choices': [choice]}) == ''


@pytest.mark.parametrize(
    'choices',
    [
        {'finish_reason': 'stop', 'message': {'content': 'not-a-list'}},
        [
            {'finish_reason': 'stop', 'message': {'content': 'first'}},
            {'finish_reason': 'stop', 'message': {'content': 'second'}},
        ],
    ],
    ids=['malformed', 'multiple'],
)
def test_malformed_or_multiple_chat_completion_choices_are_rejected(context_compaction_module, choices):
    module, _ = context_compaction_module

    assert module._response_text({'choices': choices}) == ''


def test_responses_api_accepts_only_assistant_output_text(context_compaction_module):
    module, _ = context_compaction_module
    response = {
        'status': 'completed',
        'output': [
            {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'Do not persist me.'}]},
            {'type': 'compaction', 'encrypted_content': 'opaque-provider-state'},
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': _framed(module, 'Responses summary.')}],
            },
        ],
    }

    assert module._extract_summary(module._response_text(response)) == 'Responses summary.'


@pytest.mark.parametrize('item_type', ['mcp_list_tools', 'mcp_approval_request', 'unknown_provider_item'])
def test_unknown_responses_items_reject_framed_text(context_compaction_module, item_type):
    module, _ = context_compaction_module
    response = {
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': _framed(module, 'Must not survive.')}],
            },
            {'type': item_type},
        ],
    }

    assert module._response_text(response) == ''


@pytest.mark.parametrize(
    'extra_item',
    [
        'scalar-output-item',
        {'type': 'message', 'role': 'user', 'content': [{'type': 'output_text', 'text': 'user'}]},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'roleless'}]},
    ],
    ids=['scalar', 'user-message', 'roleless-message'],
)
def test_responses_api_rejects_non_assistant_items_alongside_framed_text(context_compaction_module, extra_item):
    module, _ = context_compaction_module
    response = {
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': _framed(module, 'Must not survive.')}],
            },
            extra_item,
        ],
    }

    assert module._response_text(response) == ''


@pytest.mark.parametrize(
    'content',
    [
        {'type': 'refusal', 'refusal': 'Cannot comply.'},
        {'type': 'unexpected_content', 'text': 'provider-specific'},
    ],
    ids=['refusal', 'unexpected'],
)
def test_responses_api_rejects_non_text_content_alongside_framed_text(context_compaction_module, content):
    module, _ = context_compaction_module
    response = {
        'status': 'completed',
        'output': [
            {
                'type': 'message',
                'role': 'assistant',
                'content': [
                    {'type': 'output_text', 'text': _framed(module, 'Must not survive.')},
                    content,
                ],
            }
        ],
    }

    assert module._response_text(response) == ''


@pytest.mark.parametrize(
    'response',
    [
        {'status': 'incomplete', 'output': []},
        {
            'status': 'completed',
            'output': [
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'framed'}]},
                {'type': 'function_call', 'name': 'lookup'},
            ],
        },
        {'status': 'completed', 'output': [{'type': 'computer_call', 'name': 'browser'}]},
        {
            'status': 'completed',
            'output': [
                {
                    'type': 'message',
                    'role': 'assistant',
                    'tool_calls': [{'id': 'call-1'}],
                    'content': [{'type': 'output_text', 'text': 'framed'}],
                }
            ],
        },
    ],
    ids=['incomplete', 'function-call-after-text', 'other-call-type', 'message-tool-calls'],
)
def test_non_final_responses_api_outputs_are_rejected(context_compaction_module, response):
    module, _ = context_compaction_module

    assert module._response_text(response) == ''


def test_summary_markers_strip_outside_reasoning_and_whitespace(context_compaction_module):
    module, _ = context_compaction_module
    text = f'analysis outside\n{_framed(module, "  final summary  ")}\ntrailing outside'

    assert module._extract_summary(text) == 'final summary'


@pytest.mark.parametrize(
    'text',
    [
        '',
        'plain summary',
        '<openwebui_context_summary>missing end',
        'missing start</openwebui_context_summary>',
        '</openwebui_context_summary>reversed<openwebui_context_summary>',
        '<openwebui_context_summary></openwebui_context_summary>',
        '<openwebui_context_summary>...</openwebui_context_summary>',
        (
            '<openwebui_context_summary>one<openwebui_context_summary>two'
            '</openwebui_context_summary></openwebui_context_summary>'
        ),
        (
            '<openwebui_context_summary>one</openwebui_context_summary>'
            '<openwebui_context_summary>two</openwebui_context_summary>'
        ),
    ],
    ids=[
        'empty',
        'unframed',
        'missing-end',
        'missing-start',
        'reversed',
        'empty-frame',
        'literal-template-echo',
        'nested',
        'duplicate',
    ],
)
def test_malformed_summary_markers_are_rejected(context_compaction_module, text):
    module, _ = context_compaction_module

    assert module._extract_summary(text) == ''


def test_literal_output_instruction_echo_is_rejected(context_compaction_module):
    module, _ = context_compaction_module

    assert module._extract_summary(module.CONTEXT_SUMMARY_OUTPUT_INSTRUCTION) == ''


def test_incompatible_task_params_are_removed_without_losing_normal_params(context_compaction_module):
    module, _ = context_compaction_module
    payload = {
        'model': 'model-1',
        'temperature': 0.2,
        'max_tokens': 512,
        'format': 'json',
        'response_format': {'type': 'json_object'},
        'tools': [{'type': 'function'}],
        'tool_choice': 'required',
        'parallel_tool_calls': True,
        'functions': [{'name': 'lookup'}],
        'function_call': {'name': 'lookup'},
        'options': {
            'temperature': 0.3,
            'num_ctx': 8192,
            'format': 'json',
            'response_format': {'type': 'json_object'},
            'tools': [{'type': 'function'}],
            'tool_choice': 'required',
            'parallel_tool_calls': True,
            'functions': [{'name': 'lookup'}],
            'function_call': {'name': 'lookup'},
        },
    }

    module._remove_incompatible_task_params(payload)

    assert payload == {
        'model': 'model-1',
        'temperature': 0.2,
        'max_tokens': 512,
        'options': {'temperature': 0.3, 'num_ctx': 8192},
    }


@pytest.mark.asyncio
async def test_generate_summary_bypasses_system_prompt(context_compaction_module, monkeypatch):
    module, chat_module = context_compaction_module
    generate = AsyncMock(
        return_value={'choices': [{'finish_reason': 'stop', 'message': {'content': _framed(module, 'Generated.')}}]}
    )
    monkeypatch.setattr(chat_module, 'generate_chat_completion', generate)
    monkeypatch.setattr(
        module.Config,
        'get_many',
        AsyncMock(return_value={'task.model.params': {}, 'chat.context_compaction.model': None}),
    )
    request = SimpleNamespace(state=SimpleNamespace(metadata={'request_id': 'request-1'}))

    summary = await module._generate_summary(
        request,
        object(),
        'model-1',
        {'model-1': {'info': {'params': {'max_tokens': 512}}}},
        [{'role': 'user', 'content': 'Old question'}],
        [{'role': 'assistant', 'content': 'Recent answer'}],
        'Previous summary',
        '',
    )

    assert summary == 'Generated.'
    call = generate.await_args
    assert call.kwargs['bypass_model_params'] is True
    assert call.kwargs['bypass_system_prompt'] is True
    assert call.kwargs['form_data']['metadata'] == {'request_id': 'request-1', 'task': 'context_compaction'}
    assert call.kwargs['form_data']['messages'][0]['content'].endswith(module.CONTEXT_SUMMARY_OUTPUT_INSTRUCTION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'response',
    [
        {'choices': [{'finish_reason': 'stop', 'message': {'content': 'unframed'}}]},
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {
                        'content': None,
                        'reasoning_content': '<openwebui_context_summary>reasoning</openwebui_context_summary>',
                    },
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'length',
                    'message': {'content': '<openwebui_context_summary>partial</openwebui_context_summary>'},
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'tool_calls',
                    'message': {'tool_calls': [{'id': 'call-1'}], 'content': None},
                }
            ]
        },
        {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'lookup'}]},
    ],
    ids=['unframed', 'reasoning-only', 'length', 'tool-call', 'responses-function-call'],
)
async def test_generate_summary_raises_for_invalid_model_output(context_compaction_module, monkeypatch, response):
    module, chat_module = context_compaction_module
    monkeypatch.setattr(chat_module, 'generate_chat_completion', AsyncMock(return_value=response))
    monkeypatch.setattr(
        module.Config,
        'get_many',
        AsyncMock(return_value={'task.model.params': {}, 'chat.context_compaction.model': None}),
    )
    request = SimpleNamespace(state=SimpleNamespace(metadata={}))

    with pytest.raises(ValueError, match='no valid summary'):
        await module._generate_summary(
            request,
            object(),
            'model-1',
            {'model-1': {'info': {'params': {}}}},
            [{'role': 'user', 'content': 'Old question'}],
            [{'role': 'assistant', 'content': 'Recent answer'}],
            'Previous summary must not be used as a fallback',
            '',
        )


@pytest.mark.asyncio
async def test_invalid_automatic_compaction_does_not_write_or_mutate_messages(context_compaction_module, monkeypatch):
    module, _ = context_compaction_module
    messages = [
        {'id': f'message-{index}', 'role': 'user' if index % 2 == 0 else 'assistant', 'content': f'text-{index}'}
        for index in range(6)
    ]
    original_messages = deepcopy(messages)
    upsert = AsyncMock()
    monkeypatch.setattr(
        module,
        '_load_config',
        AsyncMock(
            return_value={
                'enable': True,
                'token_threshold': 1,
                'token_cap': 1,
                'retention_percentage': 40,
                'prompt_template': '',
            }
        ),
    )
    monkeypatch.setattr(module, '_generate_summary', AsyncMock(side_effect=ValueError('invalid summary')))
    monkeypatch.setattr(module, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(module.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert, raising=False)

    with pytest.raises(ValueError, match='invalid summary'):
        await module.compact_messages_for_request(
            SimpleNamespace(state=SimpleNamespace(metadata={})),
            object(),
            messages,
            {'chat_id': 'chat-1'},
            'model-1',
            {'model-1': {}},
        )

    upsert.assert_not_awaited()
    assert messages == original_messages
