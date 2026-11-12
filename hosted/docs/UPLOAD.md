# Upload the refactor

1. Keep a backup of the current Space, or note commit `8256323` so you can restore it.
2. Extract the ZIP. Upload its contents to the root of the existing Hugging Face Space using Files > Add file > Upload files. Keep the folder structure. Replace existing files when asked.
3. Upload `app.py`, `README.md`, `requirements.txt`, `.gitattributes`, `.gitignore`, and the `src`, `web`, `tests`, `docs`, `data`, and `models` folders. The package contains the original detector and example data. No desktop training files are needed.
4. Commit once all files are ready. Wait for the Space to rebuild. Keep the Space on Gradio and ZeroGPU; the README front matter retains the current SDK settings.
5. Test a General rewrite, My examples, edited-output detection, highlights, copy and JSON export. Check that the JSON includes run_id, prompt_version, status and settings.
6. Start a longer rewrite and select Cancel. Then run another rewrite. Clear should also stop the previous request. A token already being computed may finish before cancellation takes effect.
7. Open two independent browser sessions and submit different drafts. Each session should receive only its own result. The second rewrite waits in the queue.

Upload the extracted contents, not the ZIP itself. The files inside `src` must remain in `src`; do not flatten folders. Existing Hugging Face secrets are settings and are not included or changed by this ZIP.

This does not upload your desktop SFT adapter or start training. Full generation is unchanged apart from cooperative cancellation and shared settings. Once deployed, the GPU lifecycle and concurrent-user behavior still need the checks above.
