#include "Course.h"

// default constructor
Course::Course() {
  course_name = "";
  grade = -1;
}

// constructor
Course::Course(std::string nm, int gr):course_name(nm) {
  grade = gr;
}

// method to get the name of the course
std::string Course::getCourseName() { return course_name; }

// method to get the grade of the course
int Course::getCourseGrade() { return grade; }
