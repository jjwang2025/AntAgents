import ast
import builtins
from itertools import zip_longest

from .utils import BASE_BUILTIN_MODULES, get_source, is_valid_name


_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    """
    检查方法是否：
    - 仅使用已定义的名称
    - 不包含本地导入（例如允许numpy但不允许local_script）
    """

    def __init__(self, class_attributes: set[str], check_imports: bool = True):
        self.undefined_names = set()
        self.imports = {}
        self.from_imports = {}
        self.assigned_names = set()
        self.arg_names = set()
        self.class_attributes = class_attributes
        self.errors = []
        self.check_imports = check_imports
        self.typing_names = {"Any"}
        self.defined_classes = set()

    def visit_arguments(self, node):
        """收集函数参数"""
        self.arg_names = {arg.arg for arg in node.args}
        if node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        if node.vararg:
            self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node):
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.visit(node.value)

    def visit_With(self, node):
        """跟踪'with'语句中的别名（如'with X as y'中的'y'）"""
        for item in node.items:
            if item.optional_vars:  # 这是'with X as y'中的'y'
                if isinstance(item.optional_vars, ast.Name):
                    self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """跟踪异常别名（如'except Exception as e'中的'e'）"""
        if node.name:  # 这是'except Exception as e'中的'e'
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """跟踪带注解的赋值"""
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        """辅助方法，处理所有类型推导式中的生成器"""
        for generator in generators:
            if isinstance(generator.target, ast.Name):
                self.assigned_names.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node):
        """跟踪列表推导式中的变量"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """跟踪字典推导式中的变量"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """跟踪集合推导式中的变量"""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        """跟踪类定义"""
        self.defined_classes.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not (
                node.id in _BUILTIN_NAMES
                or node.id in BASE_BUILTIN_MODULES
                or node.id in self.arg_names
                or node.id == "self"
                or node.id in self.class_attributes
                or node.id in self.imports
                or node.id in self.from_imports
                or node.id in self.assigned_names
                or node.id in self.typing_names
                or node.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if not (
                node.func.id in _BUILTIN_NAMES
                or node.func.id in BASE_BUILTIN_MODULES
                or node.func.id in self.arg_names
                or node.func.id == "self"
                or node.func.id in self.class_attributes
                or node.func.id in self.imports
                or node.func.id in self.from_imports
                or node.func.id in self.assigned_names
                or node.func.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


def validate_tool_attributes(cls, check_imports: bool = True) -> None:
    """
    验证Tool类是否符合以下规范：
    0. __init__的任何参数都必须有默认值。
       初始化时选择的参数不可追踪，因此无法为其重建源代码，因此任何重要参数都应定义为类属性。
    1. 关于类：
        - 类属性只能是字符串或字典
        - 类属性不能是复杂属性
    2. 关于所有类方法：
        - 导入必须来自包，而不是本地文件
        - 所有方法必须自包含

    如果遇到错误则抛出所有错误，若无错误则返回None。
    """

    class ClassLevelChecker(ast.NodeVisitor):
        def __init__(self):
            self.imported_names = set()
            self.complex_attributes = set()
            self.class_attributes = set()
            self.non_defaults = set()
            self.non_literal_defaults = set()
            self.in_method = False
            self.invalid_attributes = []

        def visit_FunctionDef(self, node):
            if node.name == "__init__":
                self._check_init_function_parameters(node)
            old_context = self.in_method
            self.in_method = True
            self.generic_visit(node)
            self.in_method = old_context

        def visit_Assign(self, node):
            if self.in_method:
                return
            # 跟踪类属性
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes.add(target.id)

            # 检查赋值是否比简单字面量更复杂
            if not all(isinstance(val, (ast.Constant, ast.Dict, ast.List, ast.Set)) for val in ast.walk(node.value)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.complex_attributes.add(target.id)

            # 检查特定的类属性
            if getattr(node.targets[0], "id", "") == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(f"Class attribute 'name' must be a constant, found '{node.value}'")
                elif not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{node.value.value}'"
                    )
                elif not is_valid_name(node.value.value):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found '{node.value.value}'"
                    )

        def _check_init_function_parameters(self, node):
            # 检查参数中的默认值
            for arg, default in reversed(list(zip_longest(reversed(node.args.args), reversed(node.args.defaults)))):
                if default is None:
                    if arg.arg != "self":
                        self.non_defaults.add(arg.arg)
                elif not isinstance(default, (ast.Constant, ast.Dict, ast.List, ast.Set)):
                    self.non_literal_defaults.add(arg.arg)

    class_level_checker = ClassLevelChecker()
    source = get_source(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    if not isinstance(class_node, ast.ClassDef):
        raise ValueError("Source code must define a class")
    class_level_checker.visit(class_node)

    errors = []
    # 检查无效的类属性
    if class_level_checker.invalid_attributes:
        errors += class_level_checker.invalid_attributes
    if class_level_checker.complex_attributes:
        errors.append(
            f"Complex attributes should be defined in __init__, not as class attributes: "
            f"{', '.join(class_level_checker.complex_attributes)}"
        )
    if class_level_checker.non_defaults:
        errors.append(
            f"Parameters in __init__ must have default values, found required parameters: "
            f"{', '.join(class_level_checker.non_defaults)}"
        )
    if class_level_checker.non_literal_defaults:
        errors.append(
            f"Parameters in __init__ must have literal default values, found non-literal defaults: "
            f"{', '.join(class_level_checker.non_literal_defaults)}"
        )

    # 对所有方法运行检查
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            method_checker = MethodChecker(class_level_checker.class_attributes, check_imports=check_imports)
            method_checker.visit(node)
            errors += [f"- {node.name}: {error}" for error in method_checker.errors]

    if errors:
        raise ValueError(f"Tool validation failed for {cls.__name__}:\n" + "\n".join(errors))
    return