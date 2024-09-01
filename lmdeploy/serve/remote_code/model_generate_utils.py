import json
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatCompletionTokenLogprob, ChatMessage,
    ChoiceLogprobs, CompletionRequest, CompletionResponse,
    CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
    EmbeddingsRequest, EncodeRequest, EncodeResponse, ErrorResponse,
    FunctionResponse, FunctionStreamResponse,GenerateRequest, GenerateResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, ToolCall, TopLogprob,
    UsageInfo,ToolCallStream,FunctionResponse)
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union
from lmdeploy.serve.openai.api_server import UnmarshalRes
from lmdeploy.model import BaseChatTemplate

def create_tooluse_stream_response_json(
        request_id: str,
        created_time:int,
        model_name: str,
        index: int,
        text: str,
        tool_calls: Optional[List[ToolCall]],
        finish_reason: Optional[str] = None,
        logprobs: Optional[LogProbs] = None) -> str:
    choice_data = ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(role='assistant', content=text,tool_calls=tool_calls),
        finish_reason=finish_reason,
        logprobs=logprobs)
    response = ChatCompletionStreamResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[choice_data]
    )
    response_json = response.model_dump_json()

    return response_json

def unmarshal_qwen2_tooluse_base_tool(res: str,
                                      request: ChatCompletionRequest,
                                      created_time: int,
                                      tmp_prefix_result: str,
                                      first_return: bool,
                                      last_return: bool,
                                      name: str,
                                      action_id: int,
                                      logprobs: int) -> UnmarshalRes:
    """
    unmarshal ke qwen2 model stream response for tool use
    """
    if res.finish_reason == 'stop':
        res.finish_reason = 'tool_calls'
    
    if '<functioncall>' in tmp_prefix_result and ' {' in tmp_prefix_result:   
        if name=="":
            name = tmp_prefix_result.split('<functioncall> ')[1].split(' {')[0]
            action_id = [tool.function.name for tool in request.tools].index(name)
    else:
        return None

    if not first_return:
        fisrt_arguments=tmp_prefix_result.split(name+' ')[1]
        tool_calls = [
            ToolCall(index=str(action_id),
                id=str(action_id),
                function=FunctionResponse(name=name,arguments=fisrt_arguments))]
        response_json = create_tooluse_stream_response_json(
                request_id=request.session_id,
                created=created_time,
                model=request.model_name,
                index=0,
                text='',
                tool_calls=tool_calls,
                finish_reason=res.finish_reason,
                logprobs=logprobs)
        first_return=True
    elif not last_return:
        if tmp_prefix_result.endswith(' </'):
            tool_response=res.response.split(' </')[0]
            tool_calls = [
                ToolCallStream(index=str(action_id),function=FunctionStreamResponse(arguments=tool_response))]
            response_json = create_tooluse_stream_response_json(
                    request_id=request.session_id,
                    created=created_time,
                    model=request.model_name,
                    index=0,
                    text='',
                    tool_calls=tool_calls,
                    finish_reason=res.finish_reason,
                    logprobs=logprobs)
            last_return=True
        else:
            # if tmp_prefix_result.endswith(' {"'):
            #     res.text='{"'+res.text
            tool_calls = [
                ToolCallStream(index=str(action_id),function=FunctionStreamResponse(arguments=res.response))]
            response_json = create_tooluse_stream_response_json(
                    request_id=request.session_id,
                    created=created_time,
                    model=request.model_name,
                    index=0,
                    text='',
                    tool_calls=tool_calls,
                    finish_reason=res.finish_reason,
                    logprobs=logprobs)
    else:
        response_json = create_tooluse_stream_response_json(
                request_id=request.session_id,
                created=created_time,
                model=request.model_name,
                index=0,
                text=None,
                tool_calls=None,
                finish_reason=res.finish_reason,
                logprobs=logprobs)
    return UnmarshalRes(response_json,first_return,last_return,name,action_id)

def parse_old_tools(tool_list):
    tool_save = []
    try:
        for tool in tool_list:
            req_keys = []
            if "required" in tool["parameters"].keys():
                req_keys = tool["parameters"]["required"]
            paras = tool["parameters"]["properties"]
            for k in paras.keys():
                if k in req_keys:
                    paras[k]["required"] = True
                else:
                    paras[k]["required"] = False
            mid = {"name":tool["name"], "description":tool["description"], "parameters":paras}
            tool_save.append(mid)
    except:
        return None
    return tool_save

class KeModelBase(BaseChatTemplate):
    """Chat template of Qwem2ToolUseBase model."""

    def __init__(
            self,
            tools="""You are a helpful assistant.You have access to the following functions:
{functions}
If a you choose to call a function ONLY reply in the following format:
{start_tag}{function_name}{parameters}{end_tag}
where:
start_tag => `<functioncall>`
function_name => name of the function
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</functioncall>`

Here is an example:
<functioncall> eaxmple_function_name {"example_arguments": "example_value"} </functioncall>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
""",  # noqa
        system='<|im_start|>system\n',
        meta_instruction='You are a helpful assistant.',
        eosys='<|im_end|>\n',
        user='<|im_start|>user\n',
        eoh='<|im_end|>\n',
        assistant='<|im_start|>assistant\n',
        eoa='<|im_end|>',
        separator='\n',
        stop_words=['<|im_end|>'],
        tool='<|im_start|>tool\n',
        **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)
        self.tools = tools
        self.tool=tool

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(user=self.user,
                       assistant=self.assistant,
                       system=self.system,
                       tool=self.tool,)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.separator,
                       system=self.eosys,
                       tool=self.eoh,)
        ret = ''
        tools=kwargs.get('tools')
        new_tools=parse_old_tools(tools)
        if new_tools is None:
            new_tools=tools
        if tools is not None:
            ret+=self.system+self.tools.replace("{functions}", "\n".join([json.dumps(x, ensure_ascii=False) for x in new_tools]))+self.eosys
        else:
            if self.meta_instruction is not None and sequence_start:
                if len(messages) and messages[0]['role'] != 'system':
                    ret += f'{self.system}{self.meta_instruction}{self.eosys}'

        # converstion_len=len(messages)
        for i, message in enumerate(messages):
            role = message['role']
            content = message['content']
            if "tool_calls" in message:
                for tool_call in message['tool_calls']:
                    content=f'<functioncall> {tool_call["function"]["name"]} {json.dumps(json.loads(tool_call["function"]["arguments"]))} </functioncall>'
                    # json.dumps({"arguments": json.loads(tool_call['function']['arguments'])})
                    # content= '<functioncall> {"name": '+tool_call["function"]["name"]}'
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        ret += f'{self.assistant}'
        return ret

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        if 'ke-model-base' in model_path.lower():
            return 'ke-model-base'
     