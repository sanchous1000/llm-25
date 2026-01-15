from langfuse import Langfuse, get_client
from config import Config, load_config


YDKJS_QA_DATASET = [
    {
        "input": "What is the difference between var, let, and const in JavaScript?",
        "expected_output": "var is function-scoped, let and const are block-scoped. const cannot be reassigned.",
        "expected_books": ["scope-closures"],
        "keywords": ["var", "let", "const", "scope", "block"]
    },
    {
        "input": "How does closure work in JavaScript?",
        "expected_output": "Closure is when a function remembers and accesses variables from its outer scope even after the outer function has finished executing.",
        "expected_books": ["scope-closures"],
        "keywords": ["closure", "scope", "function", "lexical"]
    },
    {
        "input": "What is hoisting in JavaScript?",
        "expected_output": "Hoisting is JavaScript's behavior of moving declarations to the top of their scope during compilation.",
        "expected_books": ["scope-closures"],
        "keywords": ["hoisting", "declaration", "variable"]
    },
    {
        "input": "What are the primitive types in JavaScript?",
        "expected_output": "JavaScript has 7 primitive types: string, number, bigint, boolean, undefined, symbol, and null.",
        "expected_books": ["types-grammar", "get-started"],
        "keywords": ["primitive", "type", "string", "number", "boolean"]
    },
    {
        "input": "How does prototypal inheritance work?",
        "expected_output": "Objects in JavaScript can inherit properties from other objects through the prototype chain.",
        "expected_books": ["objects-classes"],
        "keywords": ["prototype", "inheritance", "chain"]
    },
    {
        "input": "What is the this keyword in JavaScript?",
        "expected_output": "The this keyword refers to the execution context of a function, determined by how the function is called.",
        "expected_books": ["objects-classes", "scope-closures"],
        "keywords": ["this", "context", "binding"]
    },
    {
        "input": "How do JavaScript classes work under the hood?",
        "expected_output": "JavaScript classes are syntactic sugar over prototypal inheritance, using constructor functions and prototype chains.",
        "expected_books": ["objects-classes"],
        "keywords": ["class", "prototype", "constructor"]
    },
    {
        "input": "What is lexical scope?",
        "expected_output": "Lexical scope means that variable access is determined by where functions are declared in the code, not where they are called.",
        "expected_books": ["scope-closures"],
        "keywords": ["lexical", "scope", "static"]
    },
    {
        "input": "How does type coercion work in JavaScript?",
        "expected_output": "Type coercion is the automatic conversion of values from one type to another, using abstract operations like ToString and ToNumber.",
        "expected_books": ["types-grammar"],
        "keywords": ["coercion", "type", "conversion"]
    },
    {
        "input": "What is the difference between == and === in JavaScript?",
        "expected_output": "== allows type coercion before comparison, === requires same type and value (strict equality).",
        "expected_books": ["types-grammar", "get-started"],
        "keywords": ["equality", "strict", "coercion"]
    },
    {
        "input": "How do arrow functions differ from regular functions?",
        "expected_output": "Arrow functions have lexical this binding, no arguments object, cannot be used as constructors.",
        "expected_books": ["scope-closures", "get-started"],
        "keywords": ["arrow", "function", "this", "lexical"]
    },
    {
        "input": "What is the temporal dead zone?",
        "expected_output": "TDZ is the period between entering a scope and the let/const declaration, where accessing the variable throws an error.",
        "expected_books": ["scope-closures"],
        "keywords": ["temporal", "dead", "zone", "TDZ"]
    },
    {
        "input": "How does the module system work in JavaScript?",
        "expected_output": "Modules encapsulate code using closures, exposing a public API while keeping internal state private.",
        "expected_books": ["scope-closures", "get-started"],
        "keywords": ["module", "import", "export", "encapsulation"]
    },
    {
        "input": "What are iterators in JavaScript?",
        "expected_output": "Iterators are objects with a next() method that returns {value, done} for sequential access to collections.",
        "expected_books": ["get-started"],
        "keywords": ["iterator", "next", "iterable"]
    },
    {
        "input": "How do you create private properties in JavaScript classes?",
        "expected_output": "Private properties are created using the # prefix before the property name in class definitions.",
        "expected_books": ["objects-classes"],
        "keywords": ["private", "class", "#"]
    }
]


def init_langfuse(config: Config):
    """Initialize Langfuse client singleton with all credentials"""
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
    return get_client()


def create_dataset(config: Config = None):
    if config is None:
        config = load_config()
    
    langfuse = init_langfuse(config)
    
    langfuse.create_dataset(
        name=config.dataset_name,
        description="YDKJS Question-Answer dataset for RAG evaluation"
    )
    
    print(f"Created dataset: {config.dataset_name}")
    
    for i, item in enumerate(YDKJS_QA_DATASET):
        langfuse.create_dataset_item(
            dataset_name=config.dataset_name,
            input={"question": item["input"]},
            expected_output={
                "answer": item["expected_output"],
                "expected_books": item["expected_books"],
                "keywords": item["keywords"]
            },
            metadata={"index": i}
        )
        print(f"  Added item {i+1}: {item['input'][:50]}...")
    
    langfuse.flush()
    print(f"\nDataset '{config.dataset_name}' created with {len(YDKJS_QA_DATASET)} items")


def get_dataset_items(config: Config = None):
    if config is None:
        config = load_config()
    
    langfuse = init_langfuse(config)
    dataset = langfuse.get_dataset(config.dataset_name)
    return dataset.items


if __name__ == "__main__":
    create_dataset()
