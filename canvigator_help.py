"""Per-task help rendering for canvigator.py.

Two public functions:

* ``print_task_help(task_name)`` — used by ``canvigator.py <task> --help``.
* ``print_global_help(task_groups)`` — used by ``canvigator.py --help``.

``FLAG_DESCRIPTIONS`` is the single source of truth for the long-form text of
each ``--flag`` so wording can't drift between the global and per-task help.
``TASK_HELP`` holds the structured per-task data; see the schema comment above
the dict.
"""
import textwrap


# Flags that apply to every task. Rendered automatically by ``print_task_help``
# even when not listed in a per-task entry's ``flags`` list.
UNIVERSAL_FLAGS = ('--crn',)


# (short, value_label, description). ``value_label`` is None for boolean flags.
FLAG_DESCRIPTIONS = {
    '--dry-run': (
        '-d', None,
        "Preview changes without modifying Canvas (a *_dryrun_* manifest is still "
        "written so you can inspect what would have been sent).",
    ),
    '--tag': (
        '-t', None,
        "Use a cloud LLM via Ollama to add a 'keywords' column with 1-3 short "
        "topical tags per question.",
    ),
    '--all': (
        '-a', None,
        "Skip the interactive picker and run across every applicable quiz in the "
        "course.",
    ),
    '--crn': (
        '-c', '<CRN>',
        "Select course by CRN (last 5 digits of the course code); skips the "
        "course picker. Useful for cron / automated runs.",
    ),
    '--months': (
        '-m', '<N>',
        "Age threshold in months (default: 6).",
    ),
    '--reply-window-days': (
        '-w', '<N>',
        "Days to accept replies after the follow-up was sent (default: 5).",
    ),
    '--auto-grade': (
        '-g', None,
        "Skip the per-student review prompt and auto-award points_possible to "
        "every submitter.",
    ),
    '--days': (
        '-n', '<N>',
        "Lookback window in days (default: 7).",
    ),
    '--cloud-questions': (
        '-q', None,
        "Route the discussion-question step to cloud Gemini 3 with a redacted "
        "prompt (tag names + integer counts only — no transcripts or themes).",
    ),
    '--send-all': (
        '-s', None,
        "Skip the per-student [y/N] confirmation and send every message. "
        "Requires a matching --dry-run run on the same task within the last "
        "10 minutes (otherwise refuses). Mutually exclusive with --dry-run.",
    ),
}


# Per-task help. Schema for each value:
#
#   description:   str        — 2-4 sentences of plain prose.
#   prerequisites: list[str]  — commands or files that must exist first.
#   inputs:        list[str]  — what the task reads (Canvas + files).
#   outputs:       list[str]  — files written + Canvas side-effects.
#   flags:         list[str]  — non-universal flags that apply (e.g. ['--all']).
#                               Universal flags (--crn) are rendered automatically.
#   examples:      list[str]  — 1-3 representative invocations.
#   run_before:    list[str]  — tasks typically run earlier in the workflow.
#   run_after:     list[str]  — tasks typically run after this one.
#
TASK_HELP = {
    'get-activity': {
        'description': (
            "Export per-student page-view, participation, and weekly-activity "
            "data to a CSV. Course-level, no quiz selection. Useful as a "
            "baseline engagement signal for a class."
        ),
        'prerequisites': [],
        'inputs': ["Canvas course (selected interactively or via --crn)."],
        'outputs': ["data/<course>/course_activity_YYYYMMDD.csv"],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 get-activity",
        ],
        'run_before': [],
        'run_after': [],
    },

    'create-pairs': {
        'description': (
            "Create student pairings from quiz scores using a distance matrix "
            "to maximize score diversity within each pair. Reads a present_*.csv "
            "to know who is in class today; the algorithm prefers the median "
            "distance method by default."
        ),
        'prerequisites': [
            "A present_*.csv in data/<course>/ with columns name, id, present "
            "(1 = present, 0 = absent).",
            "Quiz must have student attempts on Canvas.",
        ],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
            "data/<course>/present_*.csv",
        ],
        'outputs': [
            "data/<course>/pairings_based_on_quiz<id>_YYYYMMDD.csv",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 create-pairs",
        ],
        'run_before': [],
        'run_after': ['award-bonus', 'award-bonus-partner-only'],
    },

    'award-bonus': {
        'description': (
            "Award both partner and retake bonus points for a quiz. Sets fudge "
            "points on each student's BEST attempt (highest score) and leaves a "
            "Canvas submission comment with the breakdown. Partners are detected "
            "by score + answer-timing match (union-find handles triples); "
            "retakers are detected by repeated qualifying attempts on different "
            "days. Default bonus is 15% of quiz points each (30% combined). "
            "Submission data is auto-fetched from Canvas so detection always "
            "reflects every attempt through now; if the quiz's submission CSVs "
            "were written within the last 10 minutes (e.g. by a prior dry run) "
            "they are reused instead of refetching."
        ),
        'prerequisites': [],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
        ],
        'outputs': [
            "Canvas: fudge points + submission comments on each bonus-eligible "
            "student's best attempt.",
            "data/<course>/quiz<id>_detected_partners_YYYYMMDD.csv",
            "data/<course>/quiz<id>_retake_qualified_YYYYMMDD.csv",
            "data/<course>/quiz<id>_scores_w_bonus[_dryrun]_YYYYMMDD.csv",
        ],
        'flags': ['--dry-run'],
        'examples': [
            "python canvigator.py --crn 12345 --dry-run award-bonus",
            "python canvigator.py --crn 12345 award-bonus",
        ],
        'run_before': [],
        'run_after': [],
    },

    'award-bonus-partner-only': {
        'description': (
            "Award only the partner bonus (no retake). Same partner detection "
            "as award-bonus; sets fudge points on each partnered student's best "
            "attempt. Submission data is auto-fetched (with the 10-minute "
            "cache) so detection reflects every attempt through now."
        ),
        'prerequisites': [],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
        ],
        'outputs': [
            "Canvas: fudge points + submission comments on partnered students.",
            "data/<course>/quiz<id>_detected_partners_YYYYMMDD.csv",
            "data/<course>/quiz<id>_scores_w_bonus[_dryrun]_YYYYMMDD.csv",
        ],
        'flags': ['--dry-run'],
        'examples': [
            "python canvigator.py --crn 12345 --dry-run award-bonus-partner-only",
        ],
        'run_before': [],
        'run_after': [],
    },

    'award-bonus-retake-only': {
        'description': (
            "Award only the retake bonus (no partner). Same retake detection as "
            "award-bonus; sets fudge points on each qualifying retaker's best "
            "attempt. Submission data is auto-fetched (with the 10-minute "
            "cache) so detection reflects every attempt through now."
        ),
        'prerequisites': [],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
        ],
        'outputs': [
            "Canvas: fudge points + submission comments on retake-qualified "
            "students.",
            "data/<course>/quiz<id>_retake_qualified_YYYYMMDD.csv",
            "data/<course>/quiz<id>_scores_w_bonus[_dryrun]_YYYYMMDD.csv",
        ],
        'flags': ['--dry-run'],
        'examples': [
            "python canvigator.py --crn 12345 --dry-run award-bonus-retake-only",
        ],
        'run_before': [],
        'run_after': [],
    },

    'get-quiz-submission-events': {
        'description': (
            "Export per-attempt submission and event data for a selected quiz "
            "(or every published quiz with --all), and generate per-question "
            "score histograms plus first-attempt timing and page-blur "
            "histograms. Stores raw scores (no fudge points from prior bonus "
            "runs) so re-runs are idempotent."
        ),
        'prerequisites': [],
        'inputs': [
            "Canvas quiz (selected interactively, or via --crn, or --all for "
            "every published quiz in the course).",
        ],
        'outputs': [
            "data/<course>/quiz<id>_student_analysis_YYYYMMDD.csv",
            "data/<course>/quiz<id>_all_submissions_YYYYMMDD.csv",
            "data/<course>/quiz<id>_all_subs_by_question_YYYYMMDD.csv",
            "data/<course>/quiz<id>_all_subs_and_events_YYYYMMDD.csv",
            "figures/<course>/quiz<id>_*_YYYYMMDD.png (question, timing, blur "
            "histograms)",
        ],
        'flags': ['--all'],
        'examples': [
            "python canvigator.py --crn 12345 get-quiz-submission-events",
            "python canvigator.py --crn 12345 --all get-quiz-submission-events",
        ],
        'run_before': [],
        'run_after': ['award-bonus', 'send-quiz-reminder', 'prep-class-digest'],
    },

    'get-quiz-questions': {
        'description': (
            "Export quiz metadata and question content (id, type, text, points, "
            "answers) to a CSV. Skips student data download. With --tag, adds a "
            "'keywords' column via cloud LLM and writes to a separate "
            "*_questions_w_tags_*.csv so untagged and tagged exports do not "
            "overwrite each other. The 'position' column is a 1..N enumeration "
            "in the order Canvas returns the questions, which is what students "
            "see."
        ),
        'prerequisites': [
            "With --tag: OLLAMA_API_KEY must be set in set_env.sh (cloud "
            "endpoint at https://ollama.com is used).",
        ],
        'inputs': [
            "Canvas quiz (selected interactively, or via --crn, or --all for "
            "every published quiz in the course).",
        ],
        'outputs': [
            "data/<course>/quiz<id>_questions_YYYYMMDD.csv (no --tag)",
            "data/<course>/quiz<id>_questions_w_tags_YYYYMMDD.csv (with --tag)",
        ],
        'flags': ['--tag', '--all'],
        'examples': [
            "python canvigator.py --crn 12345 --tag get-quiz-questions",
            "python canvigator.py --crn 12345 --tag --all get-quiz-questions",
        ],
        'run_before': [],
        'run_after': [
            'generate-follow-up-questions', 'send-quiz-reminder',
            'send-follow-up-question', 'analyze-media-recordings',
        ],
    },

    'create-quiz': {
        'description': (
            "Interactively create an unpublished quiz on Canvas. Per question, "
            "prompts [p]laceholder, [g]enerate w/ LLM, or [e]nd. The LLM mode "
            "turns a natural-language seed into one of seven auto-gradable "
            "Canvas question types (multiple choice, true/false, matching, "
            "fill-in-the-blank, calculated, multiple answers, multiple "
            "dropdowns) and lets you accept / regenerate / skip each draft. "
            "Regenerated drafts diverge — they don't recycle the same angle."
        ),
        'prerequisites': [
            "For [g]enerate mode only: OLLAMA_API_KEY must be set. Pure "
            "placeholder runs work without it.",
        ],
        'inputs': [
            "Canvas course (selected interactively or via --crn).",
            "Instructor input at the prompt.",
        ],
        'outputs': [
            "Canvas: a new unpublished quiz with the questions you accepted.",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 create-quiz",
        ],
        'run_before': [],
        'run_after': [],
    },

    'export-anon-data': {
        'description': (
            "Anonymize all CSVs in data/<course>/ by replacing student IDs with "
            "hashed anon_ids and removing name / sis_id columns. No Canvas API "
            "needed. Output is a zip archive plus a mapping CSV. WARNING: "
            "anonymization alone does not make this data shareable — IRB / "
            "FERPA still apply."
        ),
        'prerequisites': [
            "data/<course>/ must already contain the CSVs you want to "
            "anonymize.",
        ],
        'inputs': [
            "data/<course>/*.csv (all existing files in the course directory).",
        ],
        'outputs': [
            "data/<course>/anonymized_YYYYMMDD/ (anonymized copies of every CSV)",
            "data/<course>/anonymized_YYYYMMDD.zip",
            "data/<course>/anon_mapping_YYYYMMDD.csv (real_id <-> anon_id "
            "mapping; keep this private).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 export-anon-data",
            "python canvigator.py export-anon-data",
        ],
        'run_before': [],
        'run_after': [],
    },

    'get-gradebook': {
        'description': (
            "Export the full course gradebook for all published assignments to "
            "a CSV (one row per student-assignment pair). Course-level, no quiz "
            "selection."
        ),
        'prerequisites': [],
        'inputs': ["Canvas course (selected interactively or via --crn)."],
        'outputs': [
            "data/<course>/gradebook_YYYYMMDD.csv (columns: name, "
            "sortable_name, user_id, assignment_name, assignment_id, "
            "points_possible, grade, score).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 get-gradebook",
        ],
        'run_before': [],
        'run_after': [],
    },

    'get-roster': {
        'description': (
            "Export the full course roster (students, teachers, TAs, observers) "
            "to a CSV. Course-level, no quiz selection. Useful as a stable "
            "identity reference for scripts outside Canvigator."
        ),
        'prerequisites': [],
        'inputs': ["Canvas course (selected interactively or via --crn)."],
        'outputs': [
            "data/<course>/roster_YYYYMMDD.csv (columns: name, id, sis_id, "
            "enrollment_type, state).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 get-roster",
        ],
        'run_before': [],
        'run_after': [],
    },

    'get-conversations': {
        'description': (
            "Export Canvas conversations involving any active student in this "
            "course to a CSV (sorted newest first). Pulls the instructor's full "
            "inbox + sent folders, since Canvas conversations are not "
            "course-scoped, then filters to course participants. Useful for "
            "back-filling the conversation_id of follow-up sends made before "
            "the per-send manifest existed."
        ),
        'prerequisites': [],
        'inputs': ["Canvas course (selected interactively or via --crn)."],
        'outputs': [
            "data/<course>/conversations_YYYYMMDD.csv (columns: "
            "conversation_id, subject, last_message_at, first_message_at, "
            "message_count, workflow_state, student_ids, student_names, "
            "n_student_participants, last_message).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 get-conversations",
        ],
        'run_before': [],
        'run_after': [],
    },

    'delete-old-conversations': {
        'description': (
            "Delete Canvas conversations from the instructor's inbox / sent / "
            "archived scopes whose last_message_at is older than N months "
            "(default 6). Account-wide — does NOT require course selection. "
            "Live mode requires an explicit 'yes' confirmation at the prompt."
        ),
        'prerequisites': [],
        'inputs': [
            "All Canvas conversations across the instructor's inbox + sent + "
            "archived scopes.",
        ],
        'outputs': [
            "Canvas: deleted conversations (irreversible).",
        ],
        'flags': ['--dry-run', '--months'],
        'examples': [
            "python canvigator.py --dry-run delete-old-conversations",
            "python canvigator.py --months 12 delete-old-conversations",
        ],
        'run_before': [],
        'run_after': [],
    },

    'generate-follow-up-questions': {
        'description': (
            "Generate three candidate open-ended follow-up questions per "
            "original quiz question using a cloud LLM. For each candidate, "
            "also drafts a structured rubric, an assessment guide, and "
            "pass / fail exemplars. Output goes to *_open_ended_*.csv with "
            "selected_question=0 — the instructor reviews offline and sets "
            "exactly one row per group to selected_question=1."
        ),
        'prerequisites': [
            "get-quiz-questions --tag must have produced "
            "*_questions_w_tags_*.csv for the quiz.",
            "OLLAMA_API_KEY must be set (cloud endpoint at https://ollama.com).",
        ],
        'inputs': [
            "data/<course>/quiz<id>_questions_w_tags_*.csv (selected from a "
            "menu — driven by the local CSV, not by Canvas quiz selection).",
        ],
        'outputs': [
            "data/<course>/quiz<id>_open_ended_YYYYMMDD.csv (3 rows per "
            "original question; columns include question_mode, "
            "open_ended_question, assessment_guide, rubric_json, "
            "exemplar_pass / exemplar_fail).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 generate-follow-up-questions",
        ],
        'run_before': ['get-quiz-questions'],
        'run_after': ['send-follow-up-question'],
    },

    'send-quiz-reminder': {
        'description': (
            "Send Canvas reminder messages to students about an open quiz. "
            "Each student is classified into one of four states (no attempt / "
            "imperfect / page blur / perfect clean) and gets a tailored "
            "message; imperfect-score students get a bulleted list of the "
            "questions they missed on their most recent attempt. With --all, "
            "sends ONE consolidated message per student covering every "
            "published, future-due quiz. If the quiz's submission CSVs were "
            "written within the last 10 minutes, they are reused instead of "
            "refetching from Canvas — making the typical dry-run-then-real "
            "workflow fast on the second run."
        ),
        'prerequisites': [
            "get-quiz-questions --tag must have produced "
            "*_questions_w_tags_*.csv for the quiz (the task fails fast if "
            "missing).",
            "Quiz must be published with a future due_at.",
        ],
        'inputs': [
            "Canvas quiz (selected interactively, or via --crn, or every "
            "published future-due quiz with --all).",
            "data/<course>/quiz<id>_questions_w_tags_*.csv",
        ],
        'outputs': [
            "Canvas messages (Conversations) sent to enrolled students.",
            "data/<course>/quiz<id>_reminder_sent[_dryrun]_YYYYMMDD.csv "
            "(per-quiz manifest).",
            "With --all: data/<course>/course_reminder_sent[_dryrun]_YYYYMMDD"
            ".csv (one row per student listing every quiz reminded about).",
        ],
        'flags': ['--dry-run', '--all', '--send-all'],
        'examples': [
            "python canvigator.py --crn 12345 send-quiz-reminder",
            "python canvigator.py --crn 12345 --all --dry-run send-quiz-reminder",
            "python canvigator.py --crn 12345 --send-all send-quiz-reminder",
        ],
        'run_before': ['get-quiz-questions', 'get-quiz-submission-events'],
        'run_after': ['generate-follow-up-questions', 'send-follow-up-question'],
    },

    'send-follow-up-question': {
        'description': (
            "Send the instructor-selected open-ended follow-up question (the "
            "row marked selected_question=1 in *_open_ended_*.csv) via Canvas "
            "message to every student who has attempted the quiz. The intro "
            "sentence is tailored to whether the student got the corresponding "
            "original question right on their latest attempt ('nice job, here "
            "is a more challenging follow-up to confirm mastery') or wrong "
            "('here is a follow-up to reinforce your understanding'). "
            "Mode-aware instructions: 'explain' asks for a voice recording, "
            "'draw' asks for an attached photo. Uses force_new=True so the "
            "thread is dedicated. Reuses the quiz's submission CSVs if they "
            "were written within the last 10 minutes (e.g. just after a "
            "send-quiz-reminder run) instead of refetching from Canvas."
        ),
        'prerequisites': [
            "get-quiz-questions --tag must have produced "
            "*_questions_w_tags_*.csv.",
            "generate-follow-up-questions must have produced *_open_ended_*"
            ".csv with exactly one row per question marked "
            "selected_question=1.",
        ],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
            "data/<course>/quiz<id>_questions_w_tags_*.csv",
            "data/<course>/quiz<id>_open_ended_*.csv",
        ],
        'outputs': [
            "Canvas messages sent to students who missed the question.",
            "data/<course>/quiz<id>_followup_sent[_dryrun]_YYYYMMDD.csv "
            "(manifest with conversation_id, question_id, sent_at, "
            "question_mode columns).",
        ],
        'flags': ['--dry-run', '--send-all'],
        'examples': [
            "python canvigator.py --crn 12345 --dry-run send-follow-up-question",
            "python canvigator.py --crn 12345 send-follow-up-question",
            "python canvigator.py --crn 12345 --send-all send-follow-up-question",
        ],
        'run_before': ['generate-follow-up-questions'],
        'run_after': ['assess-replies', 'send-follow-up-assessments'],
    },

    'assess-replies': {
        'description': (
            "Refresh student follow-up replies from Canvas (downloading audio "
            "and image attachments) and assess each one with a local LLM. For "
            "'explain' mode the audio model transcribes and the main model "
            "grades the transcript; for 'draw' mode the main model grades the "
            "image directly. Each reply is run N=3 times and majority-voted "
            "(confidence='high' on agreement, 'borderline' on split). Merges "
            "into a persistent (non-dated) *_followup_assessments.csv — rows "
            "with sent_assessment=1 are preserved verbatim across re-runs."
        ),
        'prerequisites': [
            "send-follow-up-question must have produced "
            "*_followup_sent_*.csv.",
            "Local Ollama running with OLLAMA_MODEL (default gemma4:31b) and "
            "OLLAMA_AUDIO_MODEL (default gemma4:e4b) available.",
        ],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn) — used for "
            "context lookup.",
            "data/<course>/quiz<id>_followup_sent_*.csv (iterated row by row "
            "to fetch each conversation).",
            "data/<course>/quiz<id>_open_ended_*.csv (provides the rubric, "
            "assessment guide, and exemplars).",
        ],
        'outputs': [
            "data/<course>/replies/ (downloaded audio + image attachments).",
            "data/<course>/quiz<id>_followup_replies_YYYYMMDD.csv (refreshed "
            "every run).",
            "data/<course>/quiz<id>_followup_assessments.csv (persistent; "
            "instructor edits 'feedback' before send-follow-up-assessments).",
        ],
        'flags': ['--reply-window-days'],
        'examples': [
            "python canvigator.py --crn 12345 assess-replies",
            "python canvigator.py --crn 12345 -w 10 assess-replies",
        ],
        'run_before': ['send-follow-up-question'],
        'run_after': ['send-follow-up-assessments', 'prep-class-digest'],
    },

    'send-follow-up-assessments': {
        'description': (
            "For every row in the persistent *_followup_assessments.csv with "
            "sent_assessment=0 and a non-empty 'feedback' value, preview the "
            "feedback and prompt the instructor [y/N] before posting it as a "
            "reply on the existing follow-up conversation (using the row's "
            "conversation_id). Default is SKIP — only an exact 'y' sends. "
            "On a confirmed send, sets sent_assessment=1 and stamps sent_at. "
            "Skipped rows stay at sent_assessment=0 so they can be revisited. "
            "The instructor edits the 'feedback' column between assess-replies "
            "and this task — feedback already sent is never overwritten."
        ),
        'prerequisites': [
            "assess-replies must have produced "
            "*_followup_assessments.csv.",
            "Instructor has reviewed the 'feedback' column and edited any "
            "borderline / fail feedback as needed.",
        ],
        'inputs': [
            "Canvas quiz (selected interactively or via --crn).",
            "data/<course>/quiz<id>_followup_assessments.csv",
        ],
        'outputs': [
            "Canvas reply messages on each follow-up conversation thread.",
            "data/<course>/quiz<id>_followup_assessments.csv (rewritten in "
            "place with sent_assessment=1 and updated sent_at on success).",
        ],
        'flags': ['--dry-run', '--send-all'],
        'examples': [
            "python canvigator.py --crn 12345 --dry-run "
            "send-follow-up-assessments",
            "python canvigator.py --crn 12345 send-follow-up-assessments",
            "python canvigator.py --crn 12345 --send-all "
            "send-follow-up-assessments",
        ],
        'run_before': ['assess-replies'],
        'run_after': [],
    },

    'create-media-recording-assignment': {
        'description': (
            "Interactively create a Canvas assignment that accepts only "
            "media_recording submissions. Prompts for title, prompt body (HTML "
            "description), points_possible (default 1), optional ISO due_at, "
            "and a publish-now y/N (default unpublished so the instructor can "
            "review before exposure)."
        ),
        'prerequisites': [],
        'inputs': [
            "Canvas course (selected interactively or via --crn).",
            "Instructor input at the prompt.",
        ],
        'outputs': [
            "Canvas: a new media-recording assignment (unpublished by "
            "default).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 "
            "create-media-recording-assignment",
        ],
        'run_before': [],
        'run_after': ['get-media-recordings'],
    },

    'get-media-recordings': {
        'description': (
            "Fetch each student's audio submission, transcode to 16 kHz mono "
            "WAV via ffmpeg (one pass, handles Canvas's DASH playback URLs), "
            "transcribe locally with the audio model, and either prompt for a "
            "per-student grade or auto-award points_possible. Default: shows "
            "each transcript and prompts for points (Enter = full credit, a "
            "number = that value, s/skip = no grade). With --auto-grade: "
            "skips the prompt and awards points_possible. Combine "
            "--auto-grade with --dry-run to preview without mutating Canvas."
        ),
        'prerequisites': [
            "ffmpeg installed and on PATH.",
            "Local Ollama running with OLLAMA_AUDIO_MODEL (default "
            "gemma4:e4b) available.",
            "A media-recording assignment must already exist on Canvas (use "
            "create-media-recording-assignment).",
        ],
        'inputs': [
            "Canvas assignment (selected interactively from the filtered list "
            "of media-recording assignments).",
        ],
        'outputs': [
            "data/<course>/media_recordings/assignment<id>/<sid>_<subid>.wav "
            "(transcoded audio).",
            "data/<course>/assignment<id>_recordings_YYYYMMDD.csv (columns: "
            "student_id, student_name, submission_id, submitted_at, "
            "audio_path, transcript, transcribed_at, grade, graded_at).",
            "Canvas: posted_grade on each submission (skipped under --dry-run).",
        ],
        'flags': ['--auto-grade', '--dry-run'],
        'examples': [
            "python canvigator.py --crn 12345 get-media-recordings",
            "python canvigator.py --crn 12345 --auto-grade --dry-run "
            "get-media-recordings",
        ],
        'run_before': ['create-media-recording-assignment'],
        'run_after': ['analyze-media-recordings', 'prep-class-digest'],
    },

    'analyze-media-recordings': {
        'description': (
            "Run two local-LLM passes against the saved transcripts: "
            "(1) classify each transcript against the unique tags from a "
            "*_questions_w_tags_*.csv (paraphrase-tolerant), and (2) extract "
            "3-5 cohort-level recurring themes. Output is a Markdown report "
            "with a tag-grounded specialized word cloud table, the "
            "LLM-extracted themes, and a roster mapping student indices to "
            "names."
        ),
        'prerequisites': [
            "get-media-recordings must have produced "
            "assignment<id>_recordings_*.csv.",
            "A *_questions_w_tags_*.csv must exist in data/<course>/ (run "
            "get-quiz-questions --tag for the relevant quiz first).",
            "Local Ollama running with OLLAMA_MODEL (default gemma4:31b) — "
            "transcripts stay local.",
        ],
        'inputs': [
            "Canvas assignment (selected interactively).",
            "data/<course>/assignment<id>_recordings_*.csv (most recent).",
            "data/<course>/quiz<qid>_questions_w_tags_*.csv (selected from a "
            "menu).",
        ],
        'outputs': [
            "data/<course>/assignment<id>_analysis_YYYYMMDD.md (tag table + "
            "themes + roster).",
        ],
        'flags': [],
        'examples': [
            "python canvigator.py --crn 12345 analyze-media-recordings",
        ],
        'run_before': ['get-media-recordings', 'get-quiz-questions'],
        'run_after': ['prep-class-digest'],
    },

    'prep-class-digest': {
        'description': (
            "Synthesize a one-page Markdown brief on cohort gaps from the last "
            "N days. Pulls three signal sources (per-tag quiz misses, "
            "follow-up reply themes, media-recording themes) and turns the top "
            "priorities into 2-3 in-class discussion questions. Default uses "
            "local Gemma 4 with the full-fidelity prompt; with "
            "--cloud-questions, the discussion-question step routes to cloud "
            "Gemini 3 with a redacted prompt (tag names + integer counts only "
            "— no transcripts or themes leave the machine)."
        ),
        'prerequisites': [
            "data/<course>/ already populated by upstream tasks "
            "(get-quiz-questions --tag + get-quiz-submission-events for "
            "misses; assess-replies for follow-up themes; get-media-recordings "
            "for transcripts).",
            "Local Ollama running with OLLAMA_MODEL (default gemma4:31b) for "
            "the default discussion-question step.",
            "OLLAMA_API_KEY set if --cloud-questions is used.",
        ],
        'inputs': [
            "data/<course>/quiz<id>_questions_w_tags_*.csv + "
            "quiz<id>_all_subs_by_question_*.csv (joined for per-tag miss "
            "counts; both must fall within the lookback window).",
            "data/<course>/quiz<id>_followup_assessments.csv (filtered to "
            "result=fail or confidence=borderline within the window).",
            "data/<course>/assignment<id>_recordings_*.csv (transcripts "
            "within the window).",
        ],
        'outputs': [
            "data/<course>/class_digest_YYYYMMDD.md (5 sections: header, quiz "
            "performance, follow-up themes, media-recording themes, "
            "discussion questions). Empty windows short-circuit and write "
            "nothing.",
        ],
        'flags': ['--days', '--cloud-questions'],
        'examples': [
            "python canvigator.py --crn 12345 prep-class-digest",
            "python canvigator.py --crn 12345 --days 14 --cloud-questions "
            "prep-class-digest",
        ],
        'run_before': [
            'assess-replies', 'get-quiz-submission-events',
            'analyze-media-recordings',
        ],
        'run_after': [],
    },
}


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

_BODY_WIDTH = 80
_BULLET_INDENT = '  - '
_BULLET_CONT = '    '
# Wide enough for the longest label, '-w, --reply-window-days <N>' (27 chars).
_FLAG_LABEL_WIDTH = 28


def _flag_label(flag):
    """Return the rendered '-x, --flag <VAL>' label for one flag."""
    short, value_label, _ = FLAG_DESCRIPTIONS[flag]
    label = f"{short}, {flag}"
    if value_label:
        label = f"{label} {value_label}"
    return label


def _print_bullets(items):
    """Print a bulleted list with consistent wrapping."""
    for item in items:
        print(textwrap.fill(
            item,
            width=_BODY_WIDTH,
            initial_indent=_BULLET_INDENT,
            subsequent_indent=_BULLET_CONT,
        ))


def _print_flags(flags):
    """Print the Flags section. Universal flags are appended automatically."""
    rendered = list(flags)
    for uf in UNIVERSAL_FLAGS:
        if uf not in rendered:
            rendered.append(uf)
    if not rendered:
        return
    print("Flags:")
    for flag in rendered:
        label = _flag_label(flag)
        _, _, text = FLAG_DESCRIPTIONS[flag]
        cont_pad = ' ' * (2 + _FLAG_LABEL_WIDTH)
        wrapped = textwrap.wrap(text, width=_BODY_WIDTH - 2 - _FLAG_LABEL_WIDTH)
        first = wrapped[0] if wrapped else ''
        print(f"  {label:<{_FLAG_LABEL_WIDTH}}{first}")
        for cont in wrapped[1:]:
            print(f"{cont_pad}{cont}")
    print()


def print_task_help(task_name):
    """Print the per-task help block for ``task_name`` to stdout."""
    if task_name not in TASK_HELP:
        print(f"No help entry for task: {task_name}")
        return
    entry = TASK_HELP[task_name]

    print(f"canvigator.py {task_name} [OPTIONS]")
    print()
    for line in textwrap.wrap(entry['description'], width=_BODY_WIDTH,
                              initial_indent='  ', subsequent_indent='  '):
        print(line)
    print()

    for header, key in (('Prerequisites', 'prerequisites'),
                        ('Inputs', 'inputs'),
                        ('Outputs', 'outputs')):
        items = entry.get(key) or []
        if not items:
            continue
        print(f"{header}:")
        _print_bullets(items)
        print()

    _print_flags(entry.get('flags') or [])

    examples = entry.get('examples') or []
    if examples:
        print("Examples:")
        for ex in examples:
            print(f"  {ex}")
        print()

    run_before = entry.get('run_before') or []
    run_after = entry.get('run_after') or []
    if run_before or run_after:
        print("Workflow:")
        if run_before:
            print(f"  Run before this: {', '.join(run_before)}")
        if run_after:
            print(f"  Run after this:  {', '.join(run_after)}")
        print()

    print("Run 'canvigator.py --help' for the full task list.")


def print_global_help(task_groups):
    """Print the global help: usage, options, grouped task list, footer.

    ``task_groups`` is the list of ``(header, [(name, one_line_desc), ...])``
    tuples maintained in canvigator.py.
    """
    print("Usage: canvigator.py [OPTIONS] <task>")
    print()
    print("Options (short and long forms are interchangeable):")
    for flag in (
        '--dry-run', '--tag', '--all', '--crn', '--months',
        '--reply-window-days', '--auto-grade', '--days', '--cloud-questions',
        '--send-all',
    ):
        label = _flag_label(flag)
        _, _, text = FLAG_DESCRIPTIONS[flag]
        cont_pad = ' ' * (2 + _FLAG_LABEL_WIDTH)
        wrapped = textwrap.wrap(text, width=_BODY_WIDTH - 2 - _FLAG_LABEL_WIDTH)
        first = wrapped[0] if wrapped else ''
        print(f"  {label:<{_FLAG_LABEL_WIDTH}}{first}")
        for cont in wrapped[1:]:
            print(f"{cont_pad}{cont}")

    all_names = [name for _, items in task_groups for name, _ in items]
    if all_names:
        max_name = max(len(n) for n in all_names)
    else:
        max_name = 0
    for header, items in task_groups:
        print(f"\n{header}")
        for name, desc in items:
            print(f"  {name:<{max_name}}  {desc}")

    print()
    print("Run 'canvigator.py <task> --help' for detailed help on a specific "
          "task (e.g. 'canvigator.py send-quiz-reminder --help').")
