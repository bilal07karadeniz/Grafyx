"""Data models for the sample project."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    """Represents a user in the system."""

    name: str
    email: str
    age: int
    role: str = "user"

    def is_admin(self) -> bool:
        """Check if the user is an admin."""
        return self.role == "admin"

    def display_name(self) -> str:
        """Get formatted display name."""
        return f"{self.name} <{self.email}>"


@dataclass
class Product:
    """Represents a product in the catalog."""

    name: str
    price: float
    category: str
    in_stock: bool = True
    tags: list[str] = field(default_factory=list)

    def discounted_price(self, discount_percent: float) -> float:
        """Calculate price after discount."""
        if discount_percent < 0 or discount_percent > 100:
            raise ValueError("Discount must be between 0 and 100")
        return self.price * (1 - discount_percent / 100)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the product."""
        if tag not in self.tags:
            self.tags.append(tag)


class Inventory:
    """Manages product inventory."""

    def __init__(self):
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """Add a product to inventory."""
        self._products[product.name] = product

    def get_product(self, name: str) -> Optional[Product]:
        """Get a product by name."""
        return self._products.get(name)

    def list_products(self, category: Optional[str] = None) -> list[Product]:
        """List products, optionally filtered by category."""
        products = list(self._products.values())
        if category:
            products = [p for p in products if p.category == category]
        return products

    def in_stock_count(self) -> int:
        """Count products currently in stock."""
        return sum(1 for p in self._products.values() if p.in_stock)


class BaseHandler:
    """Base class for all handlers."""

    def handle(self, data: dict) -> dict:
        """Process incoming data."""
        raise NotImplementedError

    def validate(self, data: dict) -> bool:
        """Validate incoming data."""
        return bool(data)


class HTTPHandler(BaseHandler):
    """Handles HTTP requests."""

    def handle(self, data: dict) -> dict:
        return {"status": 200, "body": data}


class WebSocketHandler(BaseHandler):
    """Handles WebSocket connections."""

    def handle(self, data: dict) -> dict:
        return {"type": "ws", "payload": data}


class APIHandler(HTTPHandler):
    """Handles API-specific HTTP requests. Extends HTTPHandler."""

    def handle(self, data: dict) -> dict:
        result = super().handle(data)
        result["api_version"] = "v1"
        return result
