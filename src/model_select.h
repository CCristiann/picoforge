/* model_select.h — find the model to run without typing its path.
 *
 * Models live in one directory: $PICOFORGE_MODELS, else models/ next to the picoforge binary. A model
 * is any .gguf file in it or one level below (models/NAME/NAME.gguf), named by its file stem. -m takes a
 * real path, or a short name that matches exactly one model's stem (case-insensitive substring: "q4").
 */
#ifndef PICOFORGE_MODEL_SELECT_H
#define PICOFORGE_MODEL_SELECT_H

/* The path of the model to run, owned by the caller. `arg` is the -m value, or NULL: then
 * $PICOFORGE_MODEL, then the only model found, then a numbered choice on a terminal. Dies with the
 * list of models when the choice cannot be made. */
char *model_select(const char *arg, const char *argv0);
/* `picoforge models`: one line per model (short name, size, path). */
int model_list(const char *argv0);

#endif
