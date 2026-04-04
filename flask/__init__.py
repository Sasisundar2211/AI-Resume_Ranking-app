"""Minimal Flask-compatible interface for tests."""
from __future__ import annotations
import json

class Response:
    def __init__(self, payload, status_code=200):
        self.status_code=status_code
        self._payload=payload
    def get_json(self):
        return self._payload

class _Request:
    def __init__(self):
        self._json={}
    def get_json(self, silent=True):
        return self._json

request=_Request()

class Flask:
    def __init__(self, name):
        self.routes={}
    def get(self, path):
        def deco(fn):
            self.routes[("GET",path)] = fn
            return fn
        return deco
    def post(self, path):
        def deco(fn):
            self.routes[("POST",path)] = fn
            return fn
        return deco
    def test_client(self):
        app=self
        class Client:
            def get(self, path):
                fn=app.routes.get(("GET",path))
                if not fn:
                    return Response({"error":"not found"},404)
                res=fn()
                return _normalize(res)
            def post(self, path, json=None):
                request._json = json or {}
                fn=app.routes.get(("POST",path))
                if not fn:
                    return Response({"error":"not found"},404)
                res=fn()
                return _normalize(res)
        return Client()
    def run(self, host=None, port=None, debug=None):
        return None

def _normalize(res):
    if isinstance(res, tuple):
        payload, status = res
        if isinstance(payload, Response):
            payload.status_code=status
            return payload
        return Response(payload, status)
    if isinstance(res, Response):
        return res
    return Response(res)

def jsonify(obj):
    return Response(obj,200)

def render_template(_name):
    return "ok"
