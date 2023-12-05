import numpy as np
import matplotlib.pyplot as plt


class Employee:
    def __init__(self, level, leave_rate):
        # There are 4 possible levels: E, S, M, J
        if level in ['E', 'S', 'M', 'J']:
            self.level = level
        else:
            print("The level of teh employees must be E/S/M/J.")
        # Define the lamada of the employee
        self.leave_rate = leave_rate
        self.seniority = None
        self.time_until_depart = None
        self.generate_seniority()
        self.time_until_depart = self.time_to_depart()

    def time_to_depart(self):
        # Generate a time to departure based on exponential distribution
        return np.random.exponential(1 / self.leave_rate)

    def generate_seniority(self):
        # We assume that the seniority at the beginning is a uniform and random value for different level
        seniority_base = {'E': 10, 'S': 5, 'M': 3, 'J': 1}
        seniority = np.random.uniform(0, seniority_base[self.level])
        self.seniority = seniority


class MaleEmployee(Employee):
    def __init__(self, level, leave_rate):
        super().__init__(level, leave_rate)


class FemaleEmployee(MaleEmployee):
    # Female have an additional timer kappa
    def __init__(self, level, leave_rate, kappa_rate):
        self.kappa_rate = kappa_rate
        super().__init__(level, leave_rate)

    def time_to_depart(self):
        # Generate a time to departure with additional factor for female employees
        gender_specific_time = np.random.exponential(1 / (self.leave_rate + self.kappa_rate))
        return gender_specific_time


class Company:
    def __init__(self, employee_profile, lambda_rate_set, kappa_rate_set):
        self.employees = {'E': [], 'S': [], 'M': [], 'J': []}
        # The profile is how many people in each of the level
        self.employee_profile = employee_profile
        self.lambda_rate_set = lambda_rate_set
        self.kappa_rate_set = kappa_rate_set
        # Additional attributes for tracking
        self.time = 0
        # Check how many people leave/retire
        self.vacancies = {'E': 0, 'S': 0, 'M': 0, 'J': 0}
        self.initialize_employees()  # Adjust this method for different initial states

    def initialize_employees(self):
        pass

    def next_level(self, level):
        # This function returns the next lower level in the hierarchy
        levels = {'E': 'S', 'S': 'M', 'M': 'J'}
        return levels.get(level, None)  # Returns None if there is no lower level

    def simulate_day(self):
        # Simulate a day in the company, including retirements, promotions, and hiring
        self.time += 1
        self.check_retirements()
        self.handle_promotions()
        self.hire_new_employees()

    def check_retirements(self):
        # Iterate through each level and employee to check for retirements
        for level in ['E', 'S', 'M', 'J']:
            for employee in self.employees[level]:
                employee.time_until_depart -= 1
                if employee.time_until_depart <= 0:
                    self.employees[level].remove(employee)
                    self.vacancies[level] += 1

    def handle_promotions(self):
        # Promote employees based on vacancies starting from the top level
        # J does not consider
        for level in ['E', 'S', 'M']:
            # if there is vacancy, and there is enough employees in junior level
            while self.vacancies[level] > 0 and self.employees[self.next_level(level)]:
                # Find the most senior employee in the next lower level
                most_senior = max(self.employees[self.next_level(level)], key=lambda e: e.seniority)
                self.employees[self.next_level(level)].remove(most_senior)
                most_senior.level = level  # Update the employee's level
                self.employees[level].append(most_senior)
                self.vacancies[level] -= 1
                self.vacancies[self.next_level(level)] += 1  # New vacancy in the lower level

    def hire_new_employees(self):
        # Hire new employees to fill junior-level vacancies
        while self.vacancies['J'] > 0:
            gender = np.random.choice(['male', 'female'])
            new_employee = MaleEmployee('J', self.lambda_rate_set['J']) if gender == 'male' \
                else FemaleEmployee('J', self.lambda_rate_set['J'], self.kappa_rate_set['J'])
            self.employees['J'].append(new_employee)
            self.vacancies['J'] -= 1

    def get_gender_ratios(self):
        # Calculate and return the gender ratios at each level
        gender_ratios = {}
        for level in ['E', 'S', 'M', 'J']:
            male_count = sum(1 for e in self.employees[level] if isinstance(e, MaleEmployee))
            female_count = sum(1 for e in self.employees[level] if isinstance(e, FemaleEmployee))
            total_count = male_count + female_count
            gender_ratios[level] = {'male': male_count / total_count,
                                    'female': female_count / total_count} if total_count else {'male': 0, 'female': 0}
        return gender_ratios

    def run_simulation(self, duration):
        # Data storage for plotting
        time_steps = [0]
        gender_ratios = {'E': [0], 'S': [0], 'M': [0], 'J': [0]}

        for _ in range(duration):
            self.simulate_day()
            time_steps.append(self.time)
            # Get current gender ratios
            current_ratios = self.get_gender_ratios()
            for level in current_ratios:
                gender_ratios[level].append(current_ratios[level]['female'])  # Tracking female ratio
        # Plot the results
        self.plot_gender_ratios(time_steps, gender_ratios)

    def plot_gender_ratios(self, time_steps, gender_ratios):
        pass


class Company_All_Male(Company):
    def __init__(self, employee_profile, lambda_rate_set, kappa_rate_set):
        super().__init__(employee_profile, lambda_rate_set, kappa_rate_set)
        self.initialize_employees()  # Adjust this method for different initial states

    def initialize_employees(self):
        for level in self.employees:
            for _ in range(self.employee_profile[level]):
                new_employee = MaleEmployee(level, self.lambda_rate_set[level])
                self.employees[level].append(new_employee)

    def plot_gender_ratios(self, time_steps, gender_ratios):
        plt.figure(figsize=(10, 6))
        for level in gender_ratios:
            plt.plot(time_steps, gender_ratios[level], label=f'Level {level}')

        plt.xlabel('Time')
        plt.ylabel('Female Gender Ratio')
        plt.title('Gender Ratio Trends Over Time by Level, Under All Male Initial')
        plt.legend()
        plt.show()


class Company_All_Female(Company):
    def __init__(self, employee_profile, lambda_rate_set, kappa_rate_set):
        super().__init__(employee_profile, lambda_rate_set, kappa_rate_set)
        self.initialize_employees()  # Adjust this method for different initial states

    def initialize_employees(self):
        for level in self.employees:
            for _ in range(self.employee_profile[level]):
                new_employee = FemaleEmployee(level, self.lambda_rate_set[level], self.kappa_rate_set[level])
                self.employees[level].append(new_employee)

    def plot_gender_ratios(self, time_steps, gender_ratios):
        plt.figure(figsize=(10, 6))
        for level in gender_ratios:
            plt.plot(time_steps, gender_ratios[level], label=f'Level {level}')

        plt.xlabel('Time')
        plt.ylabel('Female Gender Ratio')
        plt.title('Gender Ratio Trends Over Time by Level, Under All Female Initial')
        plt.legend()
        plt.show()


# Example usage
target_employees = {'E': 5, 'S': 20, 'M': 100, 'J': 400}
lambda_rate_set_user = {'E': 1, 'S': 1, 'M': 1, 'J': 1}
kappa_rate_set_user = {'E': 0.5, 'S': 0.5, 'M': 0.5, 'J': 0.5}

company_male = Company_All_Male(target_employees, lambda_rate_set_user, kappa_rate_set_user)

company_female = Company_All_Female(target_employees, lambda_rate_set_user, kappa_rate_set_user)

# Run the simulation for a given duration
company_male.run_simulation(duration=100)

company_female.run_simulation(duration=100)