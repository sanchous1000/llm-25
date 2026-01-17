---
title: Тестирование WebSocket
doc_id: 45c6f413cc6d
source_path: docs/advanced/testing-websockets.md
source_mtime: 2026-01-12T17:48:13.081307+00:00
---

# Тестирование WebSocket { #testing-websockets }

Вы можете использовать тот же `TestClient` для тестирования WebSocket.

Для этого используйте `TestClient` с менеджером контекста `with`, подключаясь к WebSocket:

{* ../../docs_src/app_testing/tutorial002_py39.py hl[27:31] *}

/// note | Примечание

Подробности смотрите в документации Starlette по <a href="https://www.starlette.dev/testclient/#testing-websocket-sessions" class="external-link" target="_blank">тестированию WebSocket</a>.

///
