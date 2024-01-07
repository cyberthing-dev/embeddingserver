from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI
from os import environ as env
from py_dotenv import read_dotenv
import json
from hashlib import sha1, sha3_224

read_dotenv(".env")

client = OpenAI(api_key=env["OPENAIKEY"], organization=env["OPENAIORG"])
UINT64_MAX = 2**64 - 1

try:
    with open("data/text.json", "r") as f:
        text_db: dict[str, str] = json.load(f)
except:
    text_db = {}

try:
    with open("data/texthases.npy", "rb") as f:
        text_hashes: NDArray[np.uint64] = np.load(f)
except:
    text_hashes = np.array([], dtype=np.uint64)

try:
    with open("data/embeds.npy", "rb") as f:
        embeds: NDArray[np.float64] = np.load(f)
except:
    embeds = np.zeros((1, 1536), dtype=np.float64)


class Handler(BaseHTTPRequestHandler):
    text_db = text_db
    text_hashes: NDArray[np.uint64]
    page_ids: NDArray[np.uint32]
    embeds: NDArray  # [NDArray[np.float64]]

    def log_message(self, *_, **__) -> None:
        return

    @staticmethod
    def unit_l2_normalization(vector: NDArray[np.float64]):
        """Normalize a vector using L2 normalization."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return 1 - np.dot(a, b)

    @staticmethod
    def hashedEmbed(embed: NDArray[np.float64]) -> str:
        return sha1(embed.__str__().encode("utf-8")).hexdigest()

    @staticmethod
    def textHash(text: str) -> int:
        return min(int(sha3_224(text.encode("utf-8")).hexdigest()[:16], 16), UINT64_MAX)

    def lookupEmbed(
        self,
        /,
        *,
        embed: NDArray[np.float64] | None = None,
        hashedEmbed: str | None = None,
    ) -> str | None:
        if hashedEmbed is None and embed is None:
            raise ValueError("Must provide either embed or hashedEmbed")
        elif hashedEmbed is None:
            hashedEmbed = self.hashedEmbed(embed)
        try:
            return Handler.text_db[hashedEmbed]
        except KeyError:
            return ""

    def createEmbedding(self, text: str) -> NDArray[np.float64]:
        return self.unit_l2_normalization(
            np.array(
                client.embeddings.create(input=text, model=env["EMBEDMODEL"])
                .data[0]
                .embedding,
                dtype=np.float64,
            )
        )

    def query(self, text: str):
        queryEmbed = self.createEmbedding(text)
        temp: dict[float, str] = {}
        for embed in Handler.embeds:
            distance = self.distance(queryEmbed, embed)
            lookedup = self.lookupEmbed(embed=embed)
            # print(f"{lookedup=}")
            if lookedup:
                temp[distance] = lookedup
        return [
            i
            for _, i in sorted(temp.items(), key=lambda x: x[0])
            if not i.startswith("== ")
        ][:3]

    def send(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def json(self) -> dict:
        content_len = self.headers.get("Content-Length", 0)
        if content_len == 0:
            raise ValueError("No content length")
        return json.loads(self.rfile.read(int(content_len)))

    def do_POST(self):
        try:
            if self.path == "/add":
                json = self.json()
                text: str = json["text"]
                if text.startswith("== See also ==\n") or text.endswith("=="):
                    self.send(200, {"success": True, "items": 0})
                    return
                texthash = self.textHash(text)
                if texthash in Handler.text_hashes:
                    self.send(200, {"success": True, "items": 0})
                    return
                textEmbed = self.createEmbedding(text)
                hashed = self.hashedEmbed(textEmbed)
                if not self.lookupEmbed(hashedEmbed=hashed):
                    Handler.text_db[hashed] = text
                    Handler.embeds = np.append(Handler.embeds, [textEmbed], axis=0)
                    Handler.text_hashes: NDArray[np.uint64] = np.append(
                        Handler.text_hashes, max(texthash, UINT64_MAX)
                    )

                self.send(200, {"success": True, "items": 1})

            elif self.path == "/query":
                items = self.query(self.json()["text"])
                self.send(200, {"success": True, "items": items})

            else:
                self.send(404, {"success": False, "error": "not found"})
                return
        except Exception as e:
            self.send(500, {"success": False, "error": str(e)})
            raise e


def main():
    # if you run this locally, you might want to change it to a
    # specific interface. im using * because its in a docker container
    Handler.text_db = text_db
    Handler.embeds = embeds
    Handler.text_hashes = text_hashes
    server = HTTPServer(("0.0.0.0", 4211), Handler)
    try:
        print("Hosted on http://localhost:4211")
        server.serve_forever()
    except:
        try:
            server.server_close()
        except:
            pass
        # save everything
        with open("data/text.json", "w") as f:
            json.dump(Handler.text_db, f)
        with open("data/embeds.npy", "wb") as f:
            np.save(f, Handler.embeds)
        with open("data/texthashes.npy", "wb") as f:
            np.save(f, Handler.text_hashes)


if __name__ == "__main__":
    main()
