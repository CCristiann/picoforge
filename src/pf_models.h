/* pf_models.h — find the model to run without typing its path.
 *
 * Models live in one directory: $PICOFORGE_MODELS, else models/ next to the pf binary. A model
 * is any .gguf file in it or one level below (models/NAME/NAME.gguf), named by its file stem. -m takes a
 * real path or a name, case-insensitive: the full stem, the handle `pf models` shows (the stem without
 * the words all models share: "q2", "q2-code5"), whole words of the stem ("code5"), or any unique part.
 */
#ifndef PICOFORGE_PF_MODELS_H
#define PICOFORGE_PF_MODELS_H

/* The path of the model to run, owned by the caller. `arg` is the -m value, or NULL: then
 * $PICOFORGE_MODEL, then the only model found, then a numbered choice on a terminal. Dies with the
 * list of models when the choice cannot be made. */
char *pf_models_pick(const char *arg, const char *argv0);
/* `pf models`: one line per model (short name, size, path). */
int pf_models_list(const char *argv0);

#endif
