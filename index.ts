
import express from "express";
import { JSDOM } from "jsdom";
import { marked } from "marked";
import { TiktokenModel, encoding_for_model } from "tiktoken";
import { customsearch } from "@googleapis/customsearch";
import { config } from "dotenv";
import { readFileSync } from "fs";

const PORT = 8181;
const custom_search = customsearch({
    version: "v1",
    auth: process.env.GOOGLEKEY,
    params: {
        cx: process.env.GOOGLECX
    }
});


try {
    config();
} catch (e) {
    console.log("No .env file found, using environment variables");
    console.log(e);
}

const OpenAIModel = process.env.QUICKMODEL as TiktokenModel || "gpt-3.5-turbo-16k";

const enc = encoding_for_model(OpenAIModel);

const app = express();
app.use(express.json());
app.use(express.static("static"));
app.use((req, res, next) => {

    console.log([
        new Date().toLocaleString().replace(",", ""),
        req.method,
        // fancy path
        req.path.concat((req.path.length > 8) ? "" : "\t").slice(0, 15),
        req.ip,
        req.headers.host
    ].join("\t"));

    next();
});

//const client = new OpenAI({
//    apiKey: process.env.OPENAIKEY,
//    organization: process.env.OPENAIORG
//});

/**
 * Returns an array of paragraphs and the pageIDs. To be sent to the embedder.
 */
const wikiSearch = async (query: string) => {
    const params = new URLSearchParams({
        action: "query",
        // Feed search results into...
        generator: "search",
        gsrlimit: "1",
        gsrsearch: query,
        format: "json",
        //...the extracts module
        prop: "extracts",
        exlimit: "1",
        explaintext: "true"
    });
    const baseURL = "https://en.wikipedia.org/w/api.php";
    let results: {
        [key: string]: {
            extract: string;
        }
    }
    try {
        results = (await (await fetch(
            `${baseURL}?${params}`,
            {
                method: "GET",
                headers: {
                    "Accept": "application/json",
                    // Tell them who we are in case they want to contact us
                    "User-Agent": `GPTSearch (${process.env.GHCONTACT})`,
                }
            }
        )).json()).query.pages;
    } catch (e) {
        console.log(e);
        return { error: e };
    }
    let paragraphs: string[] = [];
    const pageID = Object.keys(results)[0];
    for (const p of results[pageID].extract.split("\n\n\n")) {
        if (p.includes("== See also ==\n")) break;
        paragraphs.push(p);
    };
    return { paragraphs };
}

const googleSearch = async (query: string) => {
    const results = await custom_search.cse.list({
        cx: process.env.GOOGLECX,
        q: query,
        auth: process.env.GOOGLEKEY,
        num: 5
    });
    let snippets: string[] = [];
    let links: string[] = [];
    if (!results.data.items) return { snippets, links };
    for (const item of results.data.items) {
        snippets.push(item.snippet || "");
        links.push(item.link || "");
    }
    return {
        snippets,
        links
    };
}

class EmbedAPI {
    // Automatically determine if we're running locally or in a container
    baseURL = `http://${process.env.WSL_DISTRO_NAME != null ? "localhost" : "db"}:4211`;

    // TODO: add as batch
    add = async (texts: string[]) => {
        const response: {
            success: false;
            error: string;
        } | {
            success: true;
            items: number;
        } = await (await fetch(
            `${this.baseURL}/add`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Request-Timeout": "50"
                },
                body: JSON.stringify({ texts })
            }
        )).json();
        return response;
    }

    queryV2 = async (text: string) => {
        let links = await Promise.all([wikiSearch(text), googleSearch(text)]).then(async ([wiki, google]) => {
            await this.add([...wiki.paragraphs.slice(0, 12), ...google.snippets.slice(0, 12)]);
            return google.links;
        });

        return {
            query: await this.query(text),
            links: links
        };
    }

    /**
     * 3 most relevant items
     */
    query = async (text: string) => {
        const response: {
            success: boolean;
            items: string[] | undefined;
            error: string | undefined;
        } = await (await fetch(
            `${this.baseURL}/query`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Request-Timeout": "50"
                },
                body: JSON.stringify({ text })
            }
        )).json();
        return response;
    }
}
const embedAPI = new EmbedAPI();


app.post("/browse", async (req, res) => {

    /**
     *  Remove if you aren't deploying your own instance to OpenAI.
     *  Otherwise, you should set the environment variable CHATGPTSECRET
     * to some randomly generated key.
     */
    if (req.headers.authorization !== `Bearer ${process.env.CHATGPTSECRET}`) {
        console.log("Unauthorized request");
        res.status(401).send("Unauthorized");
        return;
    }

    const url: string = req.body.url;
    if (url.match(/https:\/\/.*\.wikipedia\.org\/.*/g))
        return res.json({
            results: {
                items: (await wikiSearch(req.body.topic)).paragraphs
            }
        });
    const topic: string = req.body.topic;

    await fetch(url).then(r => {
        return r.text();
    }).then(async (response) => {
        const window = new JSDOM(response).window;
        window.onload = async () => {
            const document = window.document;

            let paragraphs: string[] = [];
            for (const element of Array.from(
                document.querySelectorAll("p"))
                .sort((a, b) => a.textContent!.length - b.textContent!.length)
            ) {
                if (element.textContent)
                    paragraphs.push(element.textContent
                        .replace(/\n/g, " ")
                        .replace(/\t/g, " ")
                        .replace("  ", " ")
                        .replace("’", "'")
                        .replace("“", "\"")
                        .replace("”", "\"")
                        .trim()
                    );
            }
            let newParagraphs: string[] = [];
            for (const p of paragraphs) {
                let encoded = enc.encode(p);
                newParagraphs.push(
                    new TextDecoder().decode(
                        enc.decode(encoded.slice(0, 8190))
                    )
                );
            }
            await embedAPI.add(newParagraphs);

            const out = (await embedAPI.query(topic)).items || [];
            console.log(JSON.stringify(out));
            res.json(out);
        };

    });
});


app.get("/search", async (req, res) => {

    /**
     *  Remove if you aren't deploying your own instance to OpenAI.
     *  Otherwise, you should set the environment variable CHATGPTSECRET
     * to some randomly generated key.
     */
    if (req.headers.authorization !== `Bearer ${process.env.CHATGPTSECRET}`) {
        console.log("Unauthorized request");
        res.status(401).send("Unauthorized");
        return;
    }
    let result: {
        items: string[] | undefined;
        links?: string[];
    } = {
        items: (
            await embedAPI.query(req.query.q as string)
        ).items,
        links: []
    };
    if (result.items?.length === 0 || result.items === undefined) {
        let temp = await embedAPI.queryV2(req.query.q as string);
        result = {
            links: temp.links,
            items: temp.query.items
        };
    };

    const out = {
        links: result.links,
        results: result.items,
        date: new Date().toUTCString().replace(",", "").replace(" GMT", "")
    };
    console.log(JSON.stringify(out));
    res.json(out);
});

// TODO: subclass marked.Renderer to make header tags have ids
const renderer = new marked.Renderer();
renderer.heading = (text, level) => {
    const escapedText = text.toLowerCase().replace(/[^\w]+/g, "-");

    return `<h${level} id="${escapedText}"><a href="#${escapedText}">${text}</a><span>🔗</span></h${level}>`;
};

app.get("/privacy", async (_, res) => {
    const md = await marked(readFileSync("./static/privacy.md").toString(), { renderer });
    const head = `<head><title>Privacy Policy</title><link rel="stylesheet" href="/style.css"></head>`;
    res.send(`<!DOCTYPE html><html>${head}<body><main>${md}</main><center>© 2023-2024 CyberThing all rights reserved</center></body></html>`);
});

app.listen(PORT, () => {
    try {
        console.log(`Now listening on http://localhost:${PORT}`);

    } catch (e) {
        console.log(e);
    }
});
