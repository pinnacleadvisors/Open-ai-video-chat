from server.personas_store import Persona, PersonaStore


def test_upsert_and_get(tmp_path):
    store = PersonaStore(tmp_path / "db.sqlite")
    p = Persona(id="a1", name="Alice", image_path="img.png", voice="v1")
    store.upsert(p)
    out = store.get("a1")
    assert out is not None
    assert out.name == "Alice"
    assert out.voice == "v1"


def test_upsert_overwrites(tmp_path):
    store = PersonaStore(tmp_path / "db.sqlite")
    store.upsert(Persona(id="a1", name="Alice", image_path="x", voice="v1"))
    store.upsert(Persona(id="a1", name="Alice2", image_path="y", voice="v2"))
    p = store.get("a1")
    assert p.name == "Alice2"
    assert p.voice == "v2"
    assert p.image_path == "y"


def test_list_orders_newest_first(tmp_path):
    store = PersonaStore(tmp_path / "db.sqlite")
    store.upsert(Persona(id="1", name="A", image_path="x", voice="v", created_at=10))
    store.upsert(Persona(id="2", name="B", image_path="x", voice="v", created_at=20))
    store.upsert(Persona(id="3", name="C", image_path="x", voice="v", created_at=15))
    out = store.list()
    assert [p.id for p in out] == ["2", "3", "1"]


def test_delete(tmp_path):
    store = PersonaStore(tmp_path / "db.sqlite")
    store.upsert(Persona(id="a1", name="A", image_path="x", voice="v"))
    assert store.delete("a1") is True
    assert store.delete("a1") is False
    assert store.get("a1") is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "db.sqlite"
    PersonaStore(path).upsert(Persona(id="z", name="Zed", image_path="x", voice="v"))
    assert PersonaStore(path).get("z").name == "Zed"
