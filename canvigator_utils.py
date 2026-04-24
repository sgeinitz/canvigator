import os
import re
import sys
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def today_str():
    """Return today's date as a YYYYMMDD string."""
    return datetime.today().strftime('%Y%m%d')


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spin(frame, message, indent=2):
    """Write a single spinner frame with a message to stdout, overwriting the current line."""
    pad = ' ' * indent
    sys.stdout.write(f"\r{pad}{SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]} {message}  ")
    sys.stdout.flush()


def spin_done(message, indent=2):
    """Clear the spinner line and write a completion message."""
    pad = ' ' * indent
    sys.stdout.write(f"\r{pad}✓ {message}                              \n")
    sys.stdout.flush()


def setup_logging():
    """Configure logging to file and console (warnings only on console, all to file)."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"canvigator_{today_str()}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def prompt_for_index(prompt_msg, max_index):
    """Prompt the user for a 1-based numeric index, retrying on invalid input. Returns a 0-based index."""
    while True:
        raw = input(prompt_msg)
        try:
            index = int(re.sub(r'\D', '', raw))
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {max_index + 1}.")
            continue
        if index < 1 or index > max_index + 1:
            print(f"Invalid selection. Please enter a number between 1 and {max_index + 1}.")
            continue
        return index - 1


def find_latest_csv(data_path, pattern, exclude_substr=None):
    """Return the Path of the newest CSV in data_path whose name contains `pattern` and ends with _YYYYMMDD.csv.

    Files whose name contains `exclude_substr` (if given) are skipped — used to
    exclude dry-run artifacts from sent-manifest lookups. Returns None when no
    matching file exists so the caller can raise a task-specific FileNotFoundError.
    """
    matches = []
    for f in os.listdir(data_path):
        if exclude_substr and exclude_substr in f:
            continue
        m = re.search(r'(\d{8})\.csv$', f)
        if m and pattern in f:
            matches.append((m.group(1), f))

    if not matches:
        return None

    matches.sort(key=lambda t: t[0])
    return Path(data_path) / matches[-1][1]


def selectCSVFromList(directory, keyword, prompt_msg, verbose=False):
    """
    List CSV files in directory matching keyword, prompt user to select one,
    and return the full path as a Path object.
    """
    csv_files = sorted(f for f in os.listdir(directory) if keyword in f)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files containing '{keyword}' found in {directory}")

    print("\nCSV Options:")
    for i, f in enumerate(csv_files, start=1):
        fstring = f"[ {i:2d} ] {f}" if len(csv_files) > 10 else f"[ {i} ] {f}"
        print(fstring)

    csv_index = prompt_for_index(prompt_msg, len(csv_files) - 1)
    selected = csv_files[csv_index]
    print(f"\nSelected csv: {selected}")
    logger.info(f"Selected CSV: {selected}")
    return Path(directory) / selected


class CanvigatorConfig:
    """Canvigator configuration class, mostly for file paths."""

    def __init__(self):
        """Simple file path configuration for data and figures (for both input and output)."""
        self.data_path = Path.cwd() / "data"
        self.figures_path = Path.cwd() / "figures"
        self.quiz_prefix = "quiz"

    def modifyQuizPrefix(self, new_prefix):
        """Modify the quiz file name prefix."""
        self.quiz_prefix = new_prefix

    def addCourseToPath(self, course_path):
        """Change data and figures paths if necessary, creating them if they don't exist."""
        course_dir = course_path.lstrip("/")
        if self.data_path.name != course_dir:
            self.data_path = self.data_path / course_dir
            self.data_path.mkdir(parents=True, exist_ok=True)
        if self.figures_path.name != course_dir:
            self.figures_path = self.figures_path / course_dir
            self.figures_path.mkdir(parents=True, exist_ok=True)


def selectFromList(paginated_list, item_type="item"):
    """
    A general function that takes a canvasapi paginated_list object
    and lists each item in it, then prompts user to select one item.
    """
    print("\nOptions:")
    subobject_list = []
    for i, so in enumerate(paginated_list, start=1):
        print(f"[ {i:2d} ] {so}")
        subobject_list.append(so)

    if not subobject_list:
        raise ValueError(f"No {item_type}s found.")

    index = prompt_for_index(f"\nSelect {item_type} from above using index in square brackets: ", len(subobject_list) - 1)
    logger.info(f"Selected {item_type} at index {index}: {subobject_list[index]}")
    return subobject_list[index]


def selectCourseByCRN(canvas, crn):
    """Select a course by its CRN (the last 5 digits of the course code). Returns the Canvas course object."""
    for course in canvas.get_courses():
        try:
            if hasattr(course, 'course_code') and str(course.course_code).endswith(crn):
                selected = canvas.get_course(course.id)
                logger.info(f"Selected course by CRN {crn}: {selected.name}")
                return selected
        except Exception:
            continue

    print(f"Error: No course found with CRN '{crn}'.")
    sys.exit(1)


def selectCourse(canvas):
    """
    Given a canvas instance object, display a list of courses and prompt the user to select a course.
    1. First prompt the user to choose between past or current courses.
    2. List the selected type of courses (past or current).
    3. Selected course is returned as a canvas course object.
    """
    past_courses = []
    current_courses = []
    current_date = datetime.now()

    for course in canvas.get_courses():
        try:
            # Check start and end dates
            end_date = datetime.strptime(course.end_at, '%Y-%m-%dT%H:%M:%SZ') if course.end_at else None

            # Separate past and current courses
            if end_date and current_date > end_date:
                past_courses.append(course)
            elif not end_date or current_date <= end_date:
                current_courses.append(course)

        except Exception as e:
            print(f"Error: {e} - exiting")
            sys.exit(1)

    # Prompt user to select between current and past courses
    print("\nSelect Course Type:")
    print("[ 1 ] Current Courses")
    print("[ 2 ] Past Courses")
    selection = prompt_for_index("\nSelect course type (using index in '[ ]'): ", 1)

    if selection == 0:
        print("\nCurrent Courses:")
        for i, course in enumerate(current_courses, start=1):
            print(f"[ {i:2d} ] {course.name}")
        valid_courses = current_courses
    else:
        print("\nPast Courses:")
        for i, course in enumerate(past_courses, start=1):
            print(f"[ {i:2d} ] {course.name}")
        valid_courses = past_courses

    if not valid_courses:
        print("No courses found for the selected type.")
        sys.exit(1)

    course_index = prompt_for_index("\nSelect a course from the list above (using index in '[ ]'): ", len(valid_courses) - 1)

    selected = canvas.get_course(valid_courses[course_index].id)
    logger.info(f"Selected course: {selected.name}")
    return selected
