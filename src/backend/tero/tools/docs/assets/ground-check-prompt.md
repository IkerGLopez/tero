Verify if all the facts included in the RESPONSE are contained in the SOURCES.

1. If the RESPONSE is "No se" or another equivalent uncertainty statement in the response language, return the same response.
2. If any fact is not supported by the SOURCES, return "No se".
3. If the RESPONSE contains any factual statement without a Markdown citation [text](url), return "No se".
4. If all facts are supported and each fact is cited, keep the response as-is. If a code block is present, place citations outside the code block.

Only return the final response, do not include any reasoning or intermediate steps.

Example responses:

### Example Response 1

No se

### Example 2

The root of all darkness is the lack of light.

[darkness](https://en.wikipedia.org/wiki/Darkness)

---

SOURCES: {context}

RESPONSE: {response}
