import json

with open("DataBio.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("databio_export.py", "w", encoding="utf-8") as out:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            out.write(source + "\n\n")

print("Export complete.")
