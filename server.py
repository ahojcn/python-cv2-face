import sqlite3
import base64

from sanic import Sanic
from sanic.response import json

import utils

app = Sanic("faceai")


@app.route("/")
async def test(request):
    return json({"hello": "world"})


@app.route("/search_face")
async def search_face(request):
    resp = {"name": "", "image": ""}

    return json()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
