# !/bin/bash

# Create the file, set_env.sh, so that the Canvas URL and token can be set as
# environment variables.
# Once set_env.sh has been created, this
# script will not be needed any longer. Instead, when opening a terminal for the first, only 'source ./set_env.sh' will need to be called.

SETENV="set_env.sh"

if [[ -e "$SETENV" ]]; then
  echo "set_env.sh already exists (delete it if you wish to create it again, or modify it directly)"
  exit
fi

echo " ========== Creating configuration script: $SETENV ========== "

echo "Enter the URL of your Canvas LMS (oftentimes this is of the form, your_institution_name.instructure.com):"
read LMSURL

echo "Enter or paste your Canvas LMS access token here (be careful not to share this with anyone else):"
read TOKEN

echo "#!/bin/bash" > "$SETENV"
echo "export CANVAS_URL=\"$LMSURL\"" >> "$SETENV"
echo "export CANVAS_TOKEN=\"$TOKEN\"" >> "$SETENV"

# Restrict permissions so only the owner can read the token
chmod 600 "$SETENV"


DATA="data"
EXCSV="present_example.csv"
FIGS="figures"

echo "\n =========== Creating subdirectories: $DATA, $FIGS =========== "

mkdir -p $DATA $FIGS

# Create the sample .csv file with specific column names in the data/ subdirectory
echo "last_name_first_name,name,id,sis_id,email,present" > $DATA/$EXCSV
echo "\"Doe, Jane\",Jane Doe,12345,987654321,john@institution.edu,1 " >> $DATA/$EXCSV
echo "\"Doe, John\",John Doe,12346,987654322,jane@institution.edu,1 " >> $DATA/$EXCSV
echo "\"Doe, Jim\",Jim Doe,12347,987654323,jim@institution.edu,0 " >> $DATA/$EXCSV





