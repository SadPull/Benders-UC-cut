"""仅供时序操作试探的隔离工作进程；标准库、受限语法、只读输入。"""
import ast
import json
import random
import sys
from types import MappingProxyType


BUILTINS = {name: value for name, value in vars(__import__('builtins')).items()
            if name in ('abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int',
                        'len', 'list', 'max', 'min', 'range', 'reversed', 'round', 'set',
                        'sorted', 'sum', 'tuple', 'zip')}
METHODS = {'append', 'extend', 'copy', 'sort', 'index', 'count', 'get', 'items', 'keys', 'values'}
RNG_METHODS = {'random', 'randrange', 'randint', 'choice', 'sample', 'shuffle', 'uniform'}
FORBIDDEN = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
             ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await, ast.Yield,
             ast.YieldFrom, ast.Delete, ast.Try, ast.Raise)


def check_source(source):
    if not isinstance(source, str) or len(source) > 24000:
        raise ValueError('Source must be a small Python string (<=24000 characters)')
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Only one top-level function, propose(context, rng), is allowed')
    fn = tree.body[0]
    if (fn.name != 'propose' or [a.arg for a in fn.args.args] != ['context', 'rng']
            or fn.args.defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs
            or fn.args.posonlyargs):
        raise ValueError('Required interface: propose(context, rng)')
    local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    local |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    allowed = set(BUILTINS) | local | {'intervals', 'make_candidate'}
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN):
            raise ValueError('Forbidden syntax: '+type(node).__name__)
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns
                or any(a.annotation for a in node.args.args)):
            raise ValueError('Decorators and annotations are forbidden')
        if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id not in allowed):
            raise ValueError('Forbidden name: '+node.id)
        if isinstance(node, ast.Attribute):
            is_rng = isinstance(node.value, ast.Name) and node.value.id == 'rng'
            if (node.attr not in (RNG_METHODS if is_rng else METHODS)
                    or not isinstance(node.ctx, ast.Load)):
                raise ValueError('Forbidden attribute: '+node.attr)
        if isinstance(node, ast.Constant) and not isinstance(node.value, (str, int, float, bool, type(None))):
            raise ValueError('Only plain numeric/string constants are allowed')
    return tree


def intervals(row):
    """Zero-based, half-open intervals for algorithm calculations."""
    out, start = [], None
    for t, on in enumerate((*row, 0)):
        if on and start is None:
            start = t
        elif not on and start is not None:
            out.append((start, t))
            start = None
    return out


def make_candidate(base, updates):
    """updates maps zero-based unit indices to full rows; output is the existing 1-based delta."""
    return {'changes': [{'unit': i+1, 'on_intervals': [{'start': a+1, 'end': b}
            for a, b in intervals(row)]} for i, row in sorted(updates.items()) if list(row) != list(base[i])]}


def readonly(value):
    if isinstance(value, dict):
        return MappingProxyType({k: readonly(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(readonly(v) for v in value)
    return value


def deny_access(event, args):
    if (event == 'open' or event.startswith(('socket.', 'subprocess.', 'ctypes.', 'os.', 'shutil.'))
            or event in ('import', 'sys._getframe', 'sys.settrace', 'sys.setprofile')):
        raise PermissionError('Operator access denied: '+event)


def main():
    source = json.loads(sys.stdin.readline())['source']
    code = compile(check_source(source), '<frozen-operator>', 'exec')
    sys.addaudithook(deny_access)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            namespace = {'__builtins__': BUILTINS, 'intervals': intervals, 'make_candidate': make_candidate}
            exec(code, namespace)
            result = namespace['propose'](readonly(request['context']), random.Random(request['seed']))
            response = json.dumps({'candidates': result}, allow_nan=False)
            if len(response) > 200000:
                raise ValueError('Oversized operator response')
        except Exception as error:
            response = json.dumps({'error': type(error).__name__+': '+str(error)})
        print(response, flush=True)


if __name__ == '__main__':
    main()
