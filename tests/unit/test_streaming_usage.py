from optimizer.providers.usage import StreamingUsageObserver


def test_openai_stream_usage_survives_split_chunks() -> None:
    observer = StreamingUsageObserver("openai")

    observer.observe(b'data: {"choices":[],"usage":{"prompt_tokens":12,')
    observer.observe(b'"completion_tokens":3}}\n\ndata: [DONE]\n\n')

    usage = observer.finish()
    assert usage.input_tokens == 12
    assert usage.output_tokens == 3


def test_anthropic_stream_combines_start_and_delta_usage() -> None:
    observer = StreamingUsageObserver("anthropic")

    observer.observe(
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":9,'
        b'"output_tokens":1}}}\n\n'
    )
    observer.observe(b'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n')

    usage = observer.finish()
    assert usage.input_tokens == 9
    assert usage.output_tokens == 7
