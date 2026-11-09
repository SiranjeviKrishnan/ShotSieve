/*
 * ShotSieve.app main executable (release bundle).
 *
 * A tiny native launcher: it points logging at ~/Library/Logs/ShotSieve/serve.log and then
 * execv()s the embedded Python (Contents/Resources/python) running `shotsieve app`. exec matters:
 * the process LaunchServices started must itself become the window process.
 *
 * Environment (test knobs, all optional): SHOTSIEVE_DATA_DIR, SHOTSIEVE_LOG_DIR,
 * SHOTSIEVE_APP_AUTOCLOSE_S (handled by the app).
 * Build: cc -arch arm64 -O2 -mmacosx-version-min=14.0 -o ShotSieve launcher.c
 */
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>

static void mkdirs(const char *path) {
    char buf[PATH_MAX];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(buf, 0755); *p = '/'; }
    }
    mkdir(buf, 0755);
}

static const char *home_dir(void) {
    const char *h = getenv("HOME");
    if (h && *h) return h;
    struct passwd *pw = getpwuid(getuid());
    return pw ? pw->pw_dir : "/tmp";
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || !realpath(exe, real)) {
        fprintf(stderr, "ShotSieve: cannot locate itself\n");
        return 1;
    }
    char *macos = dirname(real);  /* .../Contents/MacOS */
    char res[PATH_MAX], py[PATH_MAX], icon[PATH_MAX];
    snprintf(res, sizeof res, "%s/../Resources", macos);
    snprintf(py, sizeof py, "%s/python/bin/python3.11", res);
    snprintf(icon, sizeof icon, "%s/ShotSieve.icns", res);
    if (access(py, X_OK) != 0) {
        fprintf(stderr, "ShotSieve: embedded Python missing at %s\n", py);
        return 1;
    }

    /* log folder: SHOTSIEVE_LOG_DIR, else <SHOTSIEVE_DATA_DIR>/logs, else ~/Library/Logs/ShotSieve */
    char logdir[PATH_MAX], log[PATH_MAX], old[PATH_MAX];
    const char *ld = getenv("SHOTSIEVE_LOG_DIR"), *dd = getenv("SHOTSIEVE_DATA_DIR");
    if (ld && *ld) snprintf(logdir, sizeof logdir, "%s", ld);
    else if (dd && *dd) snprintf(logdir, sizeof logdir, "%s/logs", dd);
    else snprintf(logdir, sizeof logdir, "%s/Library/Logs/ShotSieve", home_dir());
    mkdirs(logdir);
    snprintf(log, sizeof log, "%s/serve.log", logdir);
    snprintf(old, sizeof old, "%s/serve.log.1", logdir);
    struct stat st;
    if (stat(log, &st) == 0 && st.st_size > 5 * 1024 * 1024) rename(log, old);
    int fd = open(log, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); if (fd > 2) close(fd); }

    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);  /* the bundle is read-only and code-signed */
    if (!getenv("SHOTSIEVE_APP_ICON") && access(icon, R_OK) == 0) setenv("SHOTSIEVE_APP_ICON", icon, 1);

    char *args[] = {
        py, "-c",
        "import sys; from shotsieve.cli import main; sys.exit(main(['app']))",
        NULL,
    };
    execv(py, args);
    fprintf(stderr, "ShotSieve: cannot start %s: %s\n", py, strerror(errno));
    return 1;
}
