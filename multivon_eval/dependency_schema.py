"""Validate dependency snapshot structure with the existing jsonschema dependency."""
from jsonschema import Draft202012Validator

SCHEMA = 'multivon.dependencies/v1'
_TEXT = {'type': 'string', 'minLength': 1}
_HASH = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
_STRINGS = {'type': 'array', 'items': _TEXT}
_DECLARATION = {
    'type': 'object', 'required': ['version', 'dependencies', 'files'],
    'properties': {
        'version': _TEXT,
        'dependencies': {'type': 'object', 'propertyNames': _TEXT, 'additionalProperties': _TEXT},
        'files': {'type': 'object', 'propertyNames': _TEXT, 'additionalProperties': _HASH},
    },
    'additionalProperties': False,
}
_VALIDATORS = {
    'engine': Draft202012Validator({
        'type': 'object',
        'required': ['schema', 'source_digest', 'python', 'implementation', 'system', 'release', 'machine', 'packages'],
        'properties': {
            'schema': {'const': SCHEMA}, 'source_digest': _HASH, 'python': _TEXT,
            'implementation': _TEXT, 'system': _TEXT, 'release': _TEXT, 'machine': _TEXT,
            'packages': {'type': 'array', 'minItems': 1, 'items': {
                'type': 'array', 'prefixItems': [_TEXT, _TEXT], 'minItems': 2, 'maxItems': 2,
            }},
        },
        'additionalProperties': False,
    }),
    'grader': Draft202012Validator({
        'type': 'object', 'required': ['schema', 'declaration', 'opaque_fields', 'issues', 'limits'],
        'properties': {
            'schema': {'const': SCHEMA}, 'declaration': {'anyOf': [{'type': 'null'}, _DECLARATION]},
            'opaque_fields': _STRINGS, 'issues': _STRINGS, 'limits': _TEXT,
        },
        'additionalProperties': False,
    }),
}


def valid_dependency_record(value, kind):
    return _VALIDATORS[kind].is_valid(value)
