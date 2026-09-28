#!/usr/bin/python3
# -*- coding: utf-8 -*-

class JsonPropositionalLogicSchema():
    def __init__(self):
        self._head = """
{
  "allOf": [
    {"$ref": "#/definitions/_items"}%s
  ],
"""
        self._tail = """
  }
}
"""
        self._def = """  "definitions": {
    "_items": {
      "minItems": 1,
      "maxItems": 2,
      "type": "array",
      "items": {
        "anyOf": [
          {"enum": [1,0]},
          {"$ref": "#/definitions/_items"}
        ]
      }
    }""" 
        self._and = """,
    "_and": {
      "items": {
        "anyOf": [
          {"enum": [1]},
          {"$ref": "#/definitions/_items"}
        ]
      }
    }"""
        self._or = """,
    "_or": {
      "not": {
        "type": "array",
        "items": {
          "not": {
            "anyOf": [
              {"enum": [1]},
              {"$ref": "#/definitions/_items"}
            ]
          }
        }
      }
    }"""
        self._xor = """,
    "_xor": {
      "oneOf": [
        {"$ref": "#/definitions/_or"},
        {"$ref": "#/definitions/_and"}
      ]
    }"""
        self._nand = """,
    "_nand": {
      "not": {
        "type": "array",
        "items": {
          "not": {
            "anyOf": [
              {"enum": [0]},
              {"$ref": "#/definitions/_items"}
            ]
          }
        }
      }
    }"""
        self._nor = """,
    "_nor": {
      "items": {
        "anyOf": [
          {"enum": [0]},
          {"$ref": "#/definitions/_items"}
        ]
      }
    }"""
        self._xnor = """,
    "_xnor": {
      "not": {
        "$ref": "#/definitions/_xor"
      }
    }"""

    def get(self, _schema='', _xor=False,_or=False,_and=False,_xnor=False,_nor=False,_nand=False):
        json = self._head % (',\n    '+_schema if len(_schema) else '')
        json += self._def
        if _xor:
            json += self._xor
        if _or:
            json += self._or
        if _and:
            json += self._and
        if _xnor:
            json += self._xnor
        if _nor:
            json += self._nor
        if _nand:
            json += self._nand
        json += self._tail
        return json