#include "Student.h"

// Constructor
Student::Student(std::string nm, std::string no):name(nm), number(no) {
  no_of_courses = 0;
}

// Method to return student's name
std::string Student::getStuName() { return name; }

// Method to return student's number
std::string Student::getStuNumber() { return number; }

// Method to assign a grade to course
void Student::assignGrade(std::string co, int gr) {
  // check whether the maximum number of courses have been taken
  if (no_of_courses == MAXNUM) {
    std::cout << "You have exceeded the maximum number of courses !\n"; 
    return;
  }
  // create a new course
  Course c(co, gr);
  course_grades[no_of_courses++] = c;
}

// Method to return the grade of a course
int Student::getGrade(std::string co) {
  int i = 0;
  
  while (i < no_of_courses) {
    //check if course name the same as co
    if (course_grades[i].getCourseName() == co)  
      return (course_grades[i].getCourseGrade());
    i++;
  }
  return(-1);
}
