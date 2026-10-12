/* Pinned HF template, revision de4b8e4d43b917e7706784d8bb445c9af86a3540.
 * Used only to admit the specialized renderer; never interpreted at runtime. */
#include "qwen38_a_internal.h"
#include <math.h>
#include <locale.h>
#include <xlocale.h>
#include <inttypes.h>
static const char *const template_profile[] = {
    "{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n    {%- if content is string %}\n ",
    "       {{- content }}\n    {%- elif content is iterable and content is not mapping %}\n        {%- for item in content %}\n            {%- if 'image' in item or 'image_url' in item or item.type == 'image",
    "' %}\n                {%- if is_system_content %}\n                    {{- raise_exception('System message cannot contain images.') }}\n                {%- endif %}\n                {%- if do_vision_count",
    " %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Picture ' ~ image_count.value",
    " ~ ': ' }}\n                {%- endif %}\n                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\n            {%- elif 'video' in item or item.type == 'video' %}\n                {%- if is_s",
    "ystem_content %}\n                    {{- raise_exception('System message cannot contain videos.') }}\n                {%- endif %}\n                {%- if do_vision_count %}\n                    {%- set ",
    "video_count.value = video_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Video ' ~ video_count.value ~ ': ' }}\n                {%- end",
    "if %}\n                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n            {%- elif 'text' in item %}\n                {{- item.text }}\n            {%- else %}\n                {{- raise_exc",
    "eption('Unexpected item type in content.') }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif content is none or content is undefined %}\n        {{- '' }}\n    {%- else %}\n        {{- raise",
    "_exception('Unexpected content type.') }}\n    {%- endif %}\n{%- endmacro %}\n{%- if not messages %}\n    {{- raise_exception('No messages provided.') }}\n{%- endif %}\n{%- set reasoning_instructions = '' %",
    "}\n{%- if enable_thinking is undefined or enable_thinking is true %}\n    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n    {%- if resolved_reasoning_effort not in ('xhigh', '",
    "medium', 'low') %}\n        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n    {%- endif %}\n    {%- if resolved_re",
    "asoning_effort == 'xhigh' %}\n        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternati",
    "ves, and prioritize correctness, consistency, and clarity in the final answer.' %}\n    {%- elif resolved_reasoning_effort == 'low' %}\n        {%- set reasoning_instructions = 'Reasoning effort is set ",
    "to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n    {%- endif %}\n{%- endif %}\n{%- if tools and tools is iterable and tools is not m",
    "apping %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if reasoning_instructions %}\n        {{- reasoning_instructions + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou have access to the following fu",
    "nctions:\\n\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\" }}\n    {{- '\\n\\nIf you choose to call a function ONLY reply",
    " in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value fo",
    "r the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></fu",
    "nction> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE th",
    "e function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n  ",
    "  {%- if messages[0].role == 'system' %}\n        {%- set content = render_content(messages[0].content, false, true)|trim %}\n        {%- if content %}\n            {{- '\\n\\n' + content }}\n        {%- en",
    "dif %}\n    {%- endif %}\n    {{- '<|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {%- set content = render_content(messages[0].content, false, true)|trim %}\n        {%- ",
    "if content %}\n            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}\n        {%- elif reasoning_instructions %}\n   ",
    "         {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- elif reasoning_instructions %}\n        {{- '<|im_start|>system\\n' + reasoning_instructions",
    " + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messa",
    "ges|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" %}\n        {%- set content = render_content(message.content, false)|trim %}\n        {%- if not(content.startsw",
    "ith('<tool_response>') and content.endswith('</tool_response>')) %}\n            {%- set ns.multi_step_tool = false %}\n            {%- set ns.last_query_index = index %}\n        {%- endif %}\n    {%- en",
    "dif %}\n{%- endfor %}\n{%- if ns.multi_step_tool %}\n    {{- raise_exception('No user query found in messages.') }}\n{%- endif %}\n{%- for message in messages %}\n    {%- set content = render_content(messag",
    "e.content, true)|trim %}\n    {%- if message.role == \"system\" %}\n        {%- if not loop.first %}\n            {{- raise_exception('System message must be at the beginning.') }}\n        {%- endif %}\n   ",
    " {%- elif message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = ",
    "'' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- endif %}\n        {%- set reasoning_content = reasoning_cont",
    "ent|trim %}\n        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning",
    "_content + '\\n</think>\\n\\n' + content }}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls and message.tool_c",
    "alls is iterable and message.tool_calls is not mapping %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if tool_call.function is defined %}\n                    {%- set too",
    "l_call = tool_call.function %}\n                {%- endif %}\n                {%- if loop.first %}\n                    {%- if content|trim %}\n                        {{- '\\n\\n<tool_call>\\n<function=' + ",
    "tool_call.name + '>\\n' }}\n                    {%- else %}\n                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                    {%- endif %}\n                {%- else %}",
    "\n                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                {%- endif %}\n                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}\n   ",
    "                 {%- for args_name, args_value in tool_call.arguments|items %}\n                        {{- '<parameter=' + args_name + '>\\n' }}\n                        {%- set args_value = args_value ",
    "| string if args_value is string else args_value | tojson | safe %}\n                        {{- args_value }}\n                        {{- '\\n</parameter>\\n' }}\n                    {%- endfor %}\n      ",
    "          {%- endif %}\n                {{- '</function>\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%",
    "- if loop.previtem and loop.previtem.role != \"tool\" %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_resp",
    "onse>' }}\n        {%- if not loop.last and loop.nextitem.role != \"tool\" %}\n            {{- '<|im_end|>\\n' }}\n        {%- elif loop.last %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%",
    "- else %}\n        {{- raise_exception('Unexpected message role.') }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is d",
    "efined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}"
};
bool q38a_template_matches(Qwen38Bytes b) {
    size_t pos = 0;
    for (size_t i = 0; i < sizeof template_profile/sizeof *template_profile; ++i) {
        size_t n = strlen(template_profile[i]);
        if (n > b.len-pos || memcmp(b.data+pos,template_profile[i],n)) return false;
        pos += n;
    }
    return pos == b.len;
}

/* Specialized renderer of the admitted template, not a Jinja interpreter. */
typedef struct { uint8_t *data; size_t len, cap; Qwen38Status status; Qwen38Error *error; } Text;
static void fail(Text *b, Qwen38Status s, const char *detail) {
    if (!b->status) b->status = q38a_error(b->error,s,"chat",detail);
}
static void put(Text *b, Qwen38Bytes s) {
    if (b->status || !s.len) return;
    if (s.len > 16*1024*1024-b->len) { fail(b,Q38_RANGE,"16 MiB rendering limit"); return; }
    size_t total = b->len+s.len;
    if (total > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < total) cap *= 2;
        uint8_t *p = realloc(b->data,cap);
        if (!p) { fail(b,Q38_OOM,"render allocation"); return; }
        b->data = p; b->cap = cap;
    }
    memcpy(b->data+b->len,s.data,s.len); b->len = total;
}
static void literal(Text *b, const char *s) { put(b,(Qwen38Bytes){(const uint8_t *)s,strlen(s)}); }
static bool eq(Qwen38Bytes a, const char *s) { return a.len == strlen(s) && (!a.len || !memcmp(a.data,s,a.len)); }
static const Qwen38Json *member(const Qwen38Json *v, const char *key) {
    if (!v || v->kind != Q38_JSON_OBJECT) return NULL;
    for (size_t i = 0; i < v->as.object.count; ++i)
        if (eq(v->as.object.items[i].key,key)) return v->as.object.items[i].value;
    return NULL;
}
static bool string_is(const Qwen38Json *v, const char *s) { return v && v->kind == Q38_JSON_STRING && eq(v->as.string,s); }
static bool valid_string(Qwen38Bytes s) {
    if (s.len && !s.data) return false;
    for (size_t pos = 0; pos < s.len;) {
        unsigned cp; int k = q38a_utf8(s.data+pos,s.len-pos,&cp);
        if (k <= 0) return false;
        pos += (size_t)k;
    }
    return true;
}
typedef struct { const Qwen38Json *stack[64]; size_t nodes, bytes; } TreeCheck;
static bool tree_string(Text *b, TreeCheck *c, Qwen38Bytes s) {
    if (s.len > Q38A_TEXT_LIMIT-c->bytes) { fail(b,Q38_RANGE,"JSON text budget"); return false; }
    c->bytes += s.len;
    if (!valid_string(s)) { fail(b,Q38_FORMAT,"JSON string UTF-8"); return false; }
    return true;
}
static void validate_tree(Text *b, TreeCheck *c, const Qwen38Json *v, size_t depth) {
    if (b->status) return;
    if (!v) { fail(b,Q38_INVALID_ARGUMENT,"null JSON node"); return; }
    if (depth == 64 || ++c->nodes > 100000) { fail(b,Q38_RANGE,"JSON depth/node budget"); return; }
    for (size_t i = 0; i < depth; ++i)
        if (c->stack[i] == v) { fail(b,Q38_FORMAT,"cyclic JSON tree"); return; }
    c->stack[depth] = v;
    switch (v->kind) {
    case Q38_JSON_NULL: case Q38_JSON_BOOL: case Q38_JSON_INT: break;
    case Q38_JSON_FLOAT: if (!isfinite(v->as.number)) fail(b,Q38_NONFINITE,"JSON number"); break;
    case Q38_JSON_STRING: tree_string(b,c,v->as.string); break;
    case Q38_JSON_ARRAY:
        if (v->as.array.count > 100000 || (v->as.array.count && !v->as.array.items)) {
            fail(b,Q38_RANGE,"JSON array"); break;
        }
        for (size_t i = 0; i < v->as.array.count && !b->status; ++i)
            validate_tree(b,c,v->as.array.items+i,depth+1);
        break;
    case Q38_JSON_OBJECT:
        if (v->as.object.count > 4096 || (v->as.object.count && !v->as.object.items)) {
            fail(b,Q38_RANGE,"JSON object"); break;
        }
        for (size_t i = 0; i < v->as.object.count && !b->status; ++i) {
            const Qwen38JsonMember *m = v->as.object.items+i;
            if (!tree_string(b,c,m->key)) break;
            for (size_t j = 0; j < i; ++j) {
                Qwen38Bytes k = v->as.object.items[j].key;
                if (k.len == m->key.len && (!k.len || !memcmp(k.data,m->key.data,k.len)))
                    fail(b,Q38_FORMAT,"duplicate JSON key");
            }
            validate_tree(b,c,m->value,depth+1);
        }
        break;
    default: fail(b,Q38_TYPE,"JSON kind");
    }
}
static void quoted_string(Text *b, Qwen38Bytes s) {
    literal(b,"\"");
    for (size_t i = 0; i < s.len; ++i) {
        uint8_t x = s.data[i];
        switch (x) {
        case '"': literal(b,"\\\""); break;
        case '\\': literal(b,"\\\\"); break;
        case '\b': literal(b,"\\b"); break;
        case '\f': literal(b,"\\f"); break;
        case '\n': literal(b,"\\n"); break;
        case '\r': literal(b,"\\r"); break;
        case '\t': literal(b,"\\t"); break;
        default:
            if (x < 32) { char escape[7]; snprintf(escape,sizeof escape,"\\u%04x",x); literal(b,escape); }
            else put(b,(Qwen38Bytes){s.data+i,1});
        }
    }
    literal(b,"\"");
}
/* Python repr uses the shortest decimal round-trip, fixed for exponents
 * [-4,15], scientific otherwise. libc rounds each candidate in a private C
 * locale: no global setlocale, no dependence on a caller's decimal separator. */
static void json_float(Text *b, double x) {
    locale_t loc = newlocale(LC_NUMERIC_MASK,"C",NULL);
    if (!loc) { fail(b,Q38_OOM,"numeric locale"); return; }
    char candidate[64];
    for (int precision = 1; precision <= 17; ++precision) {
        snprintf_l(candidate,sizeof candidate,loc,"%.*e",precision-1,x);
        if (strtod_l(candidate,NULL,loc) == x) break;
    }
    freelocale(loc);
    char *exponent = strchr(candidate,'e');
    if (!exponent) { fail(b,Q38_FORMAT,"float serialization"); return; }
    int exp = (int)strtol(exponent+1,NULL,10);
    char digits[20]; size_t n = 0;
    const char *p = candidate;
    if (*p == '-') { literal(b,"-"); ++p; }
    while (p < exponent) { if (*p != '.') digits[n++] = *p; ++p; }
    if (exp < -4 || exp >= 16) {
        put(b,(Qwen38Bytes){(const uint8_t *)digits,1});
        if (n > 1) { literal(b,"."); put(b,(Qwen38Bytes){(const uint8_t *)digits+1,n-1}); }
        literal(b,exponent);
    } else if (exp >= 0) {
        size_t whole = (size_t)exp+1;
        for (size_t i = 0; i < whole; ++i)
            if (i < n) put(b,(Qwen38Bytes){(const uint8_t *)digits+i,1}); else literal(b,"0");
        literal(b,".");
        if (n > whole) put(b,(Qwen38Bytes){(const uint8_t *)digits+whole,n-whole}); else literal(b,"0");
    } else {
        literal(b,"0.");
        for (int i = -1; i > exp; --i) literal(b,"0");
        put(b,(Qwen38Bytes){(const uint8_t *)digits,n});
    }
}
static void json_value(Text *b, const Qwen38Json *v) {
    if (b->status) return;
    switch (v->kind) {
    case Q38_JSON_NULL: literal(b,"null"); break;
    case Q38_JSON_BOOL: literal(b,v->as.boolean ? "true" : "false"); break;
    case Q38_JSON_INT: { char num[32]; snprintf(num,sizeof num,"%" PRId64,v->as.integer); literal(b,num); break; }
    case Q38_JSON_FLOAT: json_float(b,v->as.number); break;
    case Q38_JSON_STRING: quoted_string(b,v->as.string); break;
    case Q38_JSON_ARRAY:
        literal(b,"[");
        for (size_t i = 0; i < v->as.array.count; ++i) {
            if (i) literal(b,", ");
            json_value(b,v->as.array.items+i);
        }
        literal(b,"]"); break;
    case Q38_JSON_OBJECT:
        literal(b,"{");
        for (size_t i = 0; i < v->as.object.count; ++i) {
            if (i) literal(b,", ");
            quoted_string(b,v->as.object.items[i].key); literal(b,": ");
            json_value(b,v->as.object.items[i].value);
        }
        literal(b,"}"); break;
    }
}
static bool python_space(unsigned cp) {
    return (cp >= 9 && cp <= 13) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
        cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
        cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static Qwen38Bytes trim(Qwen38Bytes s) {
    size_t first = s.len, last = s.len;
    for (size_t pos = 0; pos < s.len;) {
        unsigned cp; int width = q38a_utf8(s.data+pos,s.len-pos,&cp);
        /* All strings have already passed structural UTF-8 validation. */
        if (!python_space(cp)) { if (first == s.len) first = pos; last = pos+(size_t)width; }
        pos += (size_t)width;
    }
    return (Qwen38Bytes){s.data ? s.data+first : NULL,last-first};
}
static bool fields(Text *b, const Qwen38Json *v, const char *const *allowed) {
    if (!v || v->kind != Q38_JSON_OBJECT) { fail(b,Q38_TYPE,"object schema"); return false; }
    for (size_t i = 0; i < v->as.object.count; ++i) {
        bool found = false;
        for (size_t j = 0; allowed[j]; ++j) if (eq(v->as.object.items[i].key,allowed[j])) found = true;
        if (!found) { fail(b,Q38_UNSUPPORTED,"unknown schema field"); return false; }
    }
    return true;
}
static void content(Text *b, const Qwen38Json *v) {
    if (!v || v->kind == Q38_JSON_NULL) return;
    if (v->kind == Q38_JSON_STRING) { put(b,v->as.string); return; }
    if (v->kind != Q38_JSON_ARRAY) { fail(b,Q38_TYPE,"content schema"); return; }
    static const char *const allowed[] = {"text","type",NULL};
    for (size_t i = 0; i < v->as.array.count; ++i) {
        const Qwen38Json *item = v->as.array.items+i, *type = member(item,"type");
        if (member(item,"image") || member(item,"image_url") || member(item,"video") ||
            string_is(type,"image") || string_is(type,"video")) {
            fail(b,Q38_UNSUPPORTED,"multimodal content"); return;
        }
        if (!fields(b,item,allowed)) return;
        const Qwen38Json *text = member(item,"text");
        if (!text || text->kind != Q38_JSON_STRING || (type && !string_is(type,"text"))) {
            fail(b,Q38_TYPE,"text part schema"); return;
        }
        put(b,text->as.string);
    }
}
enum Role { SYSTEM, USER, ASSISTANT, TOOL };
typedef struct { Text raw; Qwen38Bytes text, reasoning; const Qwen38Json *calls; enum Role role; } Message;
static bool call_schema(Text *b, const Qwen38Json *call) {
    static const char *const wrapper[] = {"function","id","type",NULL};
    static const char *const direct[] = {"name","arguments",NULL};
    const Qwen38Json *fn = member(call,"function");
    if (fn) { if (!fields(b,call,wrapper)) return false; call = fn; }
    if (!fields(b,call,direct)) return false;
    const Qwen38Json *name = member(call,"name"), *args = member(call,"arguments");
    if (!name || name->kind != Q38_JSON_STRING || !name->as.string.len ||
        (args && args->kind != Q38_JSON_OBJECT && !string_is(args,""))) {
        fail(b,Q38_TYPE,"function name/arguments schema"); return false;
    }
    return true;
}
static void message_schema(Text *b, const Qwen38Json *v, Message *m, size_t index) {
    static const char *const allowed[] = {"role","content","reasoning_content","tool_calls","name","tool_call_id",NULL};
    if (!fields(b,v,allowed)) return;
    const Qwen38Json *role = member(v,"role"), *reason = member(v,"reasoning_content");
    if (string_is(role,"system")) m->role = SYSTEM;
    else if (string_is(role,"user")) m->role = USER;
    else if (string_is(role,"assistant")) m->role = ASSISTANT;
    else if (string_is(role,"tool")) m->role = TOOL;
    else { fail(b,Q38_FORMAT,"message role"); return; }
    if (m->role == SYSTEM && index) { fail(b,Q38_FORMAT,"system must be first"); return; }
    m->raw.error = b->error; content(&m->raw,member(v,"content"));
    if (m->raw.status) { b->status = m->raw.status; return; }
    m->text = trim((Qwen38Bytes){m->raw.data,m->raw.len});
    if (reason && reason->kind == Q38_JSON_STRING) m->reasoning = trim(reason->as.string);
    else if (reason && reason->kind != Q38_JSON_NULL) { fail(b,Q38_TYPE,"reasoning_content schema"); return; }
    m->calls = member(v,"tool_calls");
    if (!m->calls || m->calls->kind == Q38_JSON_NULL) { m->calls = NULL; return; }
    if (m->calls->kind != Q38_JSON_ARRAY || m->role != ASSISTANT) { fail(b,Q38_TYPE,"tool_calls schema"); return; }
    for (size_t i = 0; i < m->calls->as.array.count; ++i)
        if (!call_schema(b,m->calls->as.array.items+i)) return;
}
static void tools_schema(Text *b, const Qwen38Json *tools) {
    if (!tools || tools->kind == Q38_JSON_NULL) return;
    if (tools->kind != Q38_JSON_ARRAY) { fail(b,Q38_TYPE,"tools must be an array"); return; }
    static const char *const allowed[] = {"type","function",NULL};
    static const char *const fn_fields[] = {"name","description","parameters","strict",NULL};
    for (size_t i = 0; i < tools->as.array.count; ++i) {
        const Qwen38Json *t = tools->as.array.items+i, *fn = member(t,"function");
        if (!fields(b,t,allowed) || !fields(b,fn,fn_fields)) return;
        const Qwen38Json *name = member(fn,"name"), *parameters = member(fn,"parameters");
        const Qwen38Json *description = member(fn,"description"), *strict = member(fn,"strict");
        if (!string_is(member(t,"type"),"function") || !name || name->kind != Q38_JSON_STRING ||
            !name->as.string.len || (parameters && parameters->kind != Q38_JSON_OBJECT) ||
            (description && description->kind != Q38_JSON_STRING) || (strict && strict->kind != Q38_JSON_BOOL)) {
            fail(b,Q38_TYPE,"tool definition schema"); return;
        }
    }
}
static void render_calls(Text *b, const Message *m) {
    if (!m->calls) return;
    for (size_t i = 0; i < m->calls->as.array.count; ++i) {
        const Qwen38Json *call = m->calls->as.array.items+i, *fn = member(call,"function");
        if (fn) call = fn;
        if (!i && m->text.len) literal(b,"\n\n"); else if (i) literal(b,"\n");
        literal(b,"<tool_call>\n<function="); put(b,member(call,"name")->as.string); literal(b,">\n");
        const Qwen38Json *args = member(call,"arguments");
        if (args && args->kind == Q38_JSON_OBJECT) for (size_t j = 0; j < args->as.object.count; ++j) {
            const Qwen38JsonMember *a = args->as.object.items+j;
            literal(b,"<parameter="); put(b,a->key); literal(b,">\n");
            if (a->value->kind == Q38_JSON_STRING) put(b,a->value->as.string); else json_value(b,a->value);
            literal(b,"\n</parameter>\n");
        }
        literal(b,"</function>\n</tool_call>");
    }
}
static bool tool_response(Qwen38Bytes s) {
    static const char open[] = "<tool_response>", close[] = "</tool_response>";
    return s.len >= sizeof open+sizeof close-2 && !memcmp(s.data,open,sizeof open-1) &&
        !memcmp(s.data+s.len-(sizeof close-1),close,sizeof close-1);
}

static const char reason_xhigh[] =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

static const char reason_low[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

static const char call_instructions[] =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

static void render(Text *b, const Message *m, size_t n, const Qwen38Json *tools,
                   Qwen38ChatOptions options, size_t last_query) {
    const char *reason = "";
    if (options.enable_thinking != Q38_FALSE) {
        if (options.reasoning_effort == Q38_EFFORT_OMITTED || options.reasoning_effort == Q38_EFFORT_XHIGH)
            reason = reason_xhigh;
        else if (options.reasoning_effort == Q38_EFFORT_LOW) reason = reason_low;
    }
    Qwen38Bytes system = m[0].role == SYSTEM ? m[0].text : (Qwen38Bytes){NULL,0};
    bool has_tools = tools && tools->kind == Q38_JSON_ARRAY && tools->as.array.count;
    if (has_tools || *reason || system.len) {
        literal(b,"<|im_start|>system\n"); literal(b,reason);
        if (has_tools) {
            if (*reason) literal(b,"\n\n");
            literal(b,"# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (size_t i = 0; i < tools->as.array.count; ++i) {
                literal(b,"\n"); json_value(b,tools->as.array.items+i);
            }
            literal(b,"\n</tools>"); literal(b,call_instructions);
        }
        if (system.len) {
            if (*reason || has_tools) literal(b,"\n\n");
            put(b,system);
        }
        literal(b,"<|im_end|>\n");
    }
    for (size_t i = 0; i < n; ++i) {
        const Message *msg = m+i;
        switch (msg->role) {
        case SYSTEM: break;
        case USER:
            literal(b,"<|im_start|>user\n"); put(b,msg->text); literal(b,"<|im_end|>\n"); break;
        case ASSISTANT:
            literal(b,"<|im_start|>assistant\n");
            if (options.preserve_thinking != Q38_FALSE || i > last_query) {
                literal(b,"<think>\n"); put(b,msg->reasoning); literal(b,"\n</think>\n\n");
            }
            put(b,msg->text); render_calls(b,msg); literal(b,"<|im_end|>\n"); break;
        case TOOL:
            if (i && m[i-1].role != TOOL) literal(b,"<|im_start|>user");
            literal(b,"\n<tool_response>\n"); put(b,msg->text); literal(b,"\n</tool_response>");
            if (i+1 == n || m[i+1].role != TOOL) literal(b,"<|im_end|>\n");
            break;
        }
    }
    if (options.add_generation_prompt) {
        literal(b,"<|im_start|>assistant\n");
        literal(b,options.enable_thinking == Q38_FALSE ? "<think>\n\n</think>\n\n" : "<think>\n");
    }
}
Qwen38Status qwen38_chat_render(const Qwen38Tokenizer *tokenizer, const Qwen38Json *messages,
    const Qwen38Json *tools, Qwen38ChatOptions options, uint8_t *out, size_t cap,
    size_t *needed, Qwen38Error *e) {
    if (needed) *needed = 0;
    if (!tokenizer || !messages || !needed || (!out && cap))
        return q38a_error(e,Q38_INVALID_ARGUMENT,"chat","null argument");
    if (options.enable_thinking < Q38_OMITTED || options.enable_thinking > Q38_TRUE ||
        options.preserve_thinking < Q38_OMITTED || options.preserve_thinking > Q38_TRUE ||
        (options.enable_thinking != Q38_FALSE &&
         (options.reasoning_effort < Q38_EFFORT_OMITTED || options.reasoning_effort > Q38_EFFORT_XHIGH)))
        return q38a_error(e,Q38_RANGE,"chat","option enum");
    Text b = {.error=e}; TreeCheck check = {0};
    validate_tree(&b,&check,messages,0);
    if (tools) validate_tree(&b,&check,tools,0);
    if (b.status) return b.status;
    if (messages->kind != Q38_JSON_ARRAY || !messages->as.array.count)
        return q38a_error(e,Q38_FORMAT,"chat","nonempty message array required");
    tools_schema(&b,tools);
    if (b.status) return b.status;
    size_t n = messages->as.array.count, last_query = SIZE_MAX;
    Message *m = calloc(n,sizeof *m);
    if (!m) return q38a_error(e,Q38_OOM,"chat","message scratch");
    for (size_t i = 0; i < n && !b.status; ++i) {
        message_schema(&b,messages->as.array.items+i,m+i,i);
        if (m[i].role == USER && !tool_response(m[i].text)) last_query = i;
    }
    if (last_query == SIZE_MAX) fail(&b,Q38_FORMAT,"no user query");
    if (!b.status) render(&b,m,n,tools,options,last_query);
    for (size_t i = 0; i < n; ++i) free(m[i].raw.data);
    free(m);
    if (!b.status) {
        *needed = b.len;
        if (out && cap < b.len) fail(&b,Q38_CAPACITY,"render capacity");
        else if (out && b.len) memcpy(out,b.data,b.len);
    }
    free(b.data);
    return b.status ? b.status : q38a_error(e,Q38_OK,NULL,NULL);
}
