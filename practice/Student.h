#ifndef Student_h
#define Student_h

#include <iostream>
#include <string>
#include "Course.h"

const int MAXNUM = 20;     // Maximum number of courses allowed per student

class Student {
public :
  // Constructor
  Student(std::string nm, std::string no);  
  
  // Method to return student's name
  std::string getStuName();
  
  // Method to return student's number
  std::string getStuNumber();
  
  // Method to assign a grade to a course
  void assignGrade(std::string co, int gr);
  
  // Method to return the grade of a course
  int getGrade(std::string co);
private:
  std::string name;                    // name of the student
  std::string number;                  // the student's number 
  Course course_grades[MAXNUM];        // courses taken by student
  int no_of_courses;                   // the current number of courses taken
};
#endif
