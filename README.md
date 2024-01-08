# Embedding Server

Source code for the SearchGuy GPT app (coming soon to the ChatGPT Store).

## Goals

 - Move to sveltekit
 - add stats tracking page
 - add .py functions into sveltekit


## About the webite

The web API interface runs on ExpressJS powered by a NodeJS server and is built with Typescript. The backend embedding and database logic is run on Python 3.11.

Queries start by sending some paragraphs from a dumb search, then are refined and returned by the embedding logic. The embeddings and text are cached for a week to improve performance and accuracy.
